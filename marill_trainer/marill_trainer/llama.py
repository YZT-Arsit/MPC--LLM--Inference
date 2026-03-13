from typing import Any, List, Optional, Tuple, Union
import copy

import torch
from torch import nn

import torch.distributed
import transformers
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
import pkg_resources
transformers_version = pkg_resources.get_distribution("transformers").version

from marill_trainer import llama_flash_attn_monkey_patch

# assert version.parse(transformers.__version__) == version.parse("4.29.0"), "This file is designed to work with Transformers 4.29.0. You are using " + transformers.__version__

def rank0_print(*args):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print(*args)

def get_rotary_cos_sin(rotary_emb, value_states, position_ids, kv_seq_len):
    try:
        if position_ids is not None:
            return rotary_emb(value_states, position_ids)
    except TypeError:
        pass
    return rotary_emb(value_states, seq_len=kv_seq_len)

class LlamaModelTeacher(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # calling init method of LlamaPretrainedModel, parent class of LlamaModel
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayerOutputContext(config, flash_attn=True) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # some arguments that flash-attn now expects because of student
        for l in range(config.num_hidden_layers):
            self.layers[l].self_attn.mask_heads = False
            self.layers[l].self_attn.cluster_heads = False
            self.layers[l].self_attn.retain_context_grad = False

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

LlamaModelTeacher._prepare_decoder_attention_mask = llama_flash_attn_monkey_patch._prepare_decoder_attention_mask

class LlamaForCausalLMTeacher(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModelTeacher(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class LlamaModelStudent(LlamaModel):
    def __init__(self, config: LlamaConfig):
        if config.head_config["type"] in ["Merging", "PermutedMerging"]:
            config_with_merging = copy.deepcopy(config)
            assert(config.num_attention_heads % config.head_config["factor"] == 0)
            assert(config.num_key_value_heads == config.num_attention_heads)
            # not implemented for spaced freezing
            assert(config.layer_config["type"] != "SpacedFreezing")
            config_with_merging.num_attention_heads = config.num_attention_heads // config.head_config["factor"]
            config_with_merging.num_key_value_heads = config.num_key_value_heads // config.head_config["factor"]
            rank0_print(f"Using Head Merging with factor {config.head_config['factor']}, new num_attention_heads: {config_with_merging.num_attention_heads}")
        else:
            config_with_merging = config
        # calling init method of LlamaPretrainedModel, parent class of LlamaModel
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.skip_flash_attn = config.skip_flash_attn
        if self.skip_flash_attn is True:
            rank0_print(f"flash_attn is turned off for all layers")
        self.head_dim = config_with_merging.hidden_size // config_with_merging.num_attention_heads
        use_flash_for_trainable = (self.head_dim <= 256) and not self.skip_flash_attn
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        if config.layer_config["type"] == "BottomFreezing":
            num_frozen_layers = config.layer_config["num_layers"]
            self.trainable_list = config.layer_config["trainable_list"]
            rank0_print(f"Using bottom (layer) freezing with the first {num_frozen_layers} layers frozen")
        elif config.layer_config["type"] == "SpacedFreezing":
            self.trainable_list = config.layer_config["trainable_list"]
            rank0_print(f"Using spaced (layer) freezing with the following layers trained {self.trainable_list}")
        elif config.layer_config["type"] == "Pruning":
            self.trainable_list = config.layer_config["trainable_list"]
            rank0_print(f"Using spaced (layer) freezing with pruning only keeping the following layers: {self.trainable_list}")
        elif config.layer_config["type"] == "Full":
            self.trainable_list = config.layer_config["trainable_list"]
            rank0_print(f"All layers are trained")
        else:
            raise ValueError(f"Unknown layer config: {config.layer_config}")

        if config.layer_config["type"] in ["BottomFreezing", "SpacedFreezing", "Full"]:
            # only merge heads in the trainable layers
            self.layers = nn.ModuleList(
                [
                    LlamaDecoderLayerOutputContext(config_with_merging, flash_attn=use_flash_for_trainable) if l in self.trainable_list else LlamaDecoderLayerOutputContext(config, flash_attn=not self.skip_flash_attn) 
                    for l in range(config.num_hidden_layers)
                ]
            )
        elif config.layer_config["type"] == "Pruning":
            self.layers = nn.ModuleList(
                [
                    LlamaDecoderLayerOutputContext(config_with_merging, flash_attn=use_flash_for_trainable) if l in self.trainable_list else LlamaDecoderLayerIdentity()
                    for l in range(config.num_hidden_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown layer config: {config.layer_config}")

        # Don't mask heads by default
        for l in range(config.num_hidden_layers):
            self.layers[l].self_attn.mask_heads = False
        if config.head_config["type"] in ["Pruning", "UniformPruning"]:
            to_prune = config.head_config["to_prune"]
            for layer in range(config.num_hidden_layers):
                mask = torch.ones(config.num_attention_heads)
                # to_prune from config has everything as string
                if str(layer) in to_prune:
                    for head in to_prune[str(layer)]:
                        mask[head] = 0
                if layer in self.trainable_list:
                    print(layer, mask)
                    self.layers[layer].self_attn.mask_heads = True
                    self.layers[layer].self_attn._head_mask = mask.view(1, 1, config.num_attention_heads, 1)

        # Don't cluster heads by default
        for l in range(config.num_hidden_layers):
            self.layers[l].self_attn.cluster_heads = False
        if config.head_config["type"] in ["Clustering", "EvenClustering"]:
            clusters = config.head_config["clusters"]
            for layer in range(config.num_hidden_layers):
                layer_clusters = clusters[str(layer)]
                if layer in self.trainable_list:
                    self.layers[layer].self_attn.cluster_heads = True
                    self.layers[layer].self_attn.clusters = layer_clusters

        if config.head_analysis:
            for l in range(config.num_hidden_layers):
                self.layers[l].self_attn.retain_context_grad = True
                self.layers[l].self_attn.calculate_head_similarity = True
        else:
            for l in range(config.num_hidden_layers):
                self.layers[l].self_attn.retain_context_grad = False
                self.layers[l].self_attn.calculate_head_similarity = False

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        if transformers_version >= "4.35.0":
            self._prepare_decoder_attention_mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask
        # check how many attention heads are actually in use
        # print(self.layers[0].self_attn.num_heads)

    # copied from https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/modeling_llama.py
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        standard_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        flash_attention_mask = attention_mask

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                rank0_print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        use_flash_for_trainable = (self.head_dim <= 256) and not self.skip_flash_attn
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            ############# Flash attn logic ##############
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if idx in self.trainable_list:
                att_mask = flash_attention_mask if use_flash_for_trainable else standard_attention_mask
            else:
                att_mask = flash_attention_mask if not self.skip_flash_attn else standard_attention_mask

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    att_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=att_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlamaForCausalLMStudent(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModelStudent(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()

class LlamaDecoderLayerIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        outputs = (hidden_states,)
        # just put None for attn_output; MarillTrainer will take care of it
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (None,)
        return outputs

class LlamaFlashAttention(LlamaAttention):
    pass
LlamaFlashAttention.forward = llama_flash_attn_monkey_patch.forward

from torch import nn
from transformers.activations import GELUActivation, ClassInstantier

class SiLUActivation(nn.SiLU):
    pass

class QuadActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.125*torch.square(input) + 0.25*input + 0.5

ACT2CLS = {
    "gelu": GELUActivation,
    "quad": QuadActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "default": SiLUActivation,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)

def softmax(scores, dim):
    return torch.nn.functional.softmax(scores, dim, dtype=torch.float32)

def softmax_2relu(scores, dim, eps=1e-12):
    relu = torch.nn.functional.relu(scores)
    reduce_dim = scores.shape[dim]
    out = (relu + eps/reduce_dim) / (torch.sum(relu, dim=dim, keepdims=True)+eps)
    return out

def softmax_2quad(scores, attention_mask_zero_one, dim, constant=5):
    scores =  (scores + constant) ** 2
    scores *= attention_mask_zero_one
    scores = scores / torch.sum(scores, dim=dim, keepdims=True)
    return scores

def softmax_scaling(scores, attention_mask_zero_one, dim):
    dim_len = scores.shape[dim]
    scores /= math.sqrt(dim_len)
    scores *= attention_mask_zero_one
    return scores

class Linear2Quad(nn.Module):
    __constants__ = ['num_heads', 'max_len']
    num_heads: int
    max_len: int
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, num_heads: int, max_len: int, constant=5.0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        # initializing weights with all 1s and bias with all 5s
        self.weight = torch.nn.parameter.Parameter(torch.ones((num_heads, max_len), **factory_kwargs))
        self.bias = torch.nn.parameter.Parameter(torch.full((num_heads, max_len), constant, **factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, num_heads, q_len, kv_seq_len = input.size()
        ret = ((input.transpose(1, 2) * self.weight[..., :kv_seq_len]) + self.bias[..., :kv_seq_len]).transpose(1, 2)
        assert(ret.size() == (bsz, num_heads, q_len, kv_seq_len))
        return ret

    def extra_repr(self) -> str:
        return f'max_len={self.max_len}, num_heads={self.num_heads}'

ACT2SFN = {
    "default": softmax,
    "smax": softmax,
    "2relu": softmax_2relu,
    "2quad": softmax_2quad,
    "l2quad": softmax_2quad,
    "scale": softmax_scaling,
    }

class LlamaMLPCustomAct(LlamaMLP):
    def __init__(self, config):
        super(LlamaMLP, self).__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        if transformers_version >= "4.35.0":
            self.config = config

class LlamaDecoderLayerOutputContext(nn.Module):
    def __init__(self, config: LlamaConfig, flash_attn=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        if flash_attn:
            self.self_attn = LlamaFlashAttention(config=config)
        else:
            self.self_attn = LlamaStandardAttention(config=config)
        # transformers 4.35.0 defines self.mlp this way
        self.mlp = LlamaMLPCustomAct(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

    # """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # Changed here; outputting context instead of self_attn_weights if output_attentions=True.
        if output_attentions:
            outputs += (attn_output,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

# to ensure we can use FSDP with a uniform requires_grad=True mask
class LlamaDecoderLayerTrainable(LlamaDecoderLayerOutputContext):
    pass

class LlamaDecoderLayerFrozen(LlamaDecoderLayerOutputContext):
    pass

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import math
class LlamaStandardAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.softmax_act = ACT2SFN[config.softmax_act]
        self.softmax_type = config.softmax_act
        if self.softmax_type == "l2quad":
            # input dims: (bsz, self.num_heads, q_len, kv_seq_len)
            self.linear_2quad = Linear2Quad(num_heads=self.num_heads, max_len=self.max_position_embeddings, constant=5.0)
        self.config = config

    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cluster heads according to specified medoids
        if self.cluster_heads:
            assert(self.num_key_value_heads == self.num_heads)
            assert(key_states.shape == (bsz, self.num_heads, q_len, self.head_dim))
            assert(query_states.shape == (bsz, self.num_heads, q_len, self.head_dim))
            for cluster_idx in self.clusters:
                cluster = self.clusters[cluster_idx]
                medoid_idx = cluster['medoid_idx']
                for idx in cluster['indices']:
                    key_states[:, idx] = key_states[:, medoid_idx]
                    query_states[:, idx] = query_states[:, medoid_idx]

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = get_rotary_cos_sin(self.rotary_emb, value_states, position_ids, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.to(torch.float32)
        if self.softmax_type in ["l2quad"]:
            attn_weights = self.linear_2quad(attn_weights)
        if self.softmax_type in ["2quad", "l2quad"]:
            assert torch.finfo(hidden_states.dtype).min in attention_mask and not -1e4 in attention_mask, "attention bias is different than torch.finfo(dtype).min"
            attention_mask_zero_one = torch.where(attention_mask == torch.finfo(hidden_states.dtype).min, torch.zeros_like(attention_mask).to(attention_mask.device), attention_mask)
            constant = 0 if self.softmax_type == "l2quad" else 5
            attn_weights = self.softmax_act(attn_weights, attention_mask_zero_one, dim=-1, constant=constant)  #nn.Softmax(dim=-1)(attention_scores)
        elif self.softmax_type in ["scale"]:
            assert torch.finfo(hidden_states.dtype).min in attention_mask and not -1e4 in attention_mask, "attention bias is different than torch.finfo(dtype).min"
            attention_mask_zero_one = torch.where(attention_mask == torch.finfo(hidden_states.dtype).min, torch.zeros_like(attention_mask).to(attention_mask.device), attention_mask)
            attn_weights = self.softmax_act(attn_weights, attention_mask_zero_one, dim=-1)
        elif self.softmax_type in ["2relu", "smax"]:
            attn_weights = self.softmax_act(attn_weights, dim=-1)
        else:
            raise ValueError(f"Unknown softmax type: {self.softmax_type}")
        attn_weights = attn_weights.to(query_states.dtype)

        '''old smax
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        '''

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Mask heads (if specified)
        if self.mask_heads:
            attn_output = attn_output * self._head_mask.to(device=attn_output.device, dtype=attn_output.dtype)

        # retrain context_grad for head importance calculation if config says so
        if self.retain_context_grad:
            self.context_layer_val = attn_output
            self.context_layer_val.retain_grad()
        # preserve attn_weights to calculate head_similarity
        if self.calculate_head_similarity:
            # store attn_weights before the softmax
            self.attn_weights = attn_weights

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
