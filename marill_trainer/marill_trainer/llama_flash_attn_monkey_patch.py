from typing import Any, List, Optional, Tuple

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from einops import rearrange

#from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

def get_rotary_cos_sin(rotary_emb, value_states, position_ids, kv_seq_len):
    try:
        if position_ids is not None:
            return rotary_emb(value_states, position_ids)
    except TypeError:
        pass
    return rotary_emb(value_states, seq_len=kv_seq_len)

def compat_unpad_input(x, key_padding_mask):
    unpad_result = unpad_input(x, key_padding_mask)
    if len(unpad_result) == 4:
        return unpad_result
    if len(unpad_result) == 5:
        x_unpad, indices, cu_q_lens, max_s, _ = unpad_result
        return x_unpad, indices, cu_q_lens, max_s
    raise ValueError(f"Unexpected unpad_input return length: {len(unpad_result)}")

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel
    
    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

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
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = get_rotary_cos_sin(self.rotary_emb, value_states, position_ids, kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    # Expand grouped-query key/value heads to match query heads before packing qkv.
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2) # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3) # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask


    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                device=qkv.device)
        #torch.cuda.synchronize()
        #torch.cuda.empty_cache()
        #torch.cuda.reset_peak_memory_stats()
        #_max_memory_start = torch.cuda.max_memory_allocated()
        output = flash_attn_varlen_qkvpacked_func(# flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        #torch.cuda.synchronize()
        #print(f"flash attn peak:{(torch.cuda.max_memory_allocated() - _max_memory_start) / 2 ** 20}")
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        
        x_unpad, indices, cu_q_lens, max_s = compat_unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func( #flash_attn_unpadded_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        #torch.cuda.synchronize()
        #print(f"flash attn peak:{(torch.cuda.max_memory_allocated() - _max_memory_start) / 2 ** 20}")
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                    indices, bsz, q_len),
                        'b s (h d) -> b s h d', h=nheads)

    # Mask heads (if specified)
    if self.mask_heads:
        output = output * self._head_mask.to(device=output.device, dtype=output.dtype)

    # retrain context_grad for head importance calculation if config says so
    if self.retain_context_grad:
        self.context_layer_val = output
        self.context_layer_val.retain_grad()

    return self.o_proj(rearrange(output,
                                    'b s h d -> b s (h d)')), None, None


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                    inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
