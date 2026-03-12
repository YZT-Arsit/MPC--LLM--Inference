import torch
import enum
import torch.distributed
from transformers import Trainer
import transformers
from torch.nn import MSELoss
import torch.distributed as dist
from typing import Dict
from dataclasses import dataclass, field
from typing import Dict, Optional
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import re
import wandb
import hashlib
import sys

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel
from marill_trainer.llama import rank0_print, LlamaForCausalLMStudent, LlamaForCausalLMTeacher, transformers_version
from marill_trainer.config import LayerConfig, WeightConfig, HeadConfig, ActType, SoftmaxType
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class KLMethod(enum.Enum):
    Forward = 1
    Reverse = 2
    JSD = 3

    def __init__(self, ratio = None):
        self._ratio = None

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if self in [KLMethod.JSD]:
            self._ratio = value
        else:
            raise ValueError("ratio is only set for JSD")

    def from_str(string):
        match = re.match(r'(\w+)(?:\{(\d+(\.\d+)?)\})?$', string)
        if match:
            operation = match.group(1)
            value = float(match.group(2)) if match.group(2) is not None else None
            if operation == "forward":
                assert(value is None)
                config = KLMethod.Forward
            elif operation == "reverse":
                assert(value is None)
                config = KLMethod.Reverse
            elif operation == "jsd":
                config = KLMethod.JSD
            else:
                raise ValueError(f"Invalid KLMethod: {string}")
            if value is not None:
                config.ratio = value
            return config
        else:
            raise ValueError(f"Invalid KLMethod: {string}")

class TrainingPhase(enum.Enum):
    TRAIN = 1
    HEAD_ANALYSIS = 2

    def from_str(string):
        if string == "training":
            config = TrainingPhase.TRAIN
        elif string == "analysis":
            config = TrainingPhase.HEAD_ANALYSIS
        else:
            raise ValueError(f"Invalid TrainingPhase: {string}")
        return config

@dataclass
class MarillTrainingArguments(transformers.TrainingArguments):
    teacher_model: Optional[str] = field(default=None)
    # loss coefficient for hidden, attention, logits, label
    hs_coef: Optional[float] = field(default=0.0)
    att_coef: Optional[float] = field(default=0.0)
    logit_coef: Optional[float] = field(default=0.0)
    label_coef: Optional[float] = field(default=0.0)
    layer_config: str = field(
        default="full",
        metadata = {
            "help" : "Possible options: 'full'; 'bottom{num_layers_to_freeze}'; 'spaced{num_layers_to_freeze}'; 'prune{num_layers_to_prune}'"
        }
    )
    weight_config: str = field(
        default="full",
        metadata = {
            "help" : "Possible options: 'full'; 'lora{lora_rank}'"
        }
    )
    head_config: str = field(
        default="default",
        metadata = {
            "help" : "Possible options: 'default'; 'merge{num_heads_to_merge}'; 'permuted_merge{num_heads_to_merge}'; 'cluster{head_per_layer/num_clusters}'; 'even_cluster{head_per_layer/num_clusters}'; 'prune{heads_before_pruning/heads_after_pruning}'; 'uniform_prune{heads_before_pruning/heads_after_pruning}'"
        }
    )
    act_type: str = field(
        default="default",
        metadata = {
            "choices" : ["default", "quad", "relu"]
        }
    )
    smax_type: str = field(
        default="default",
        metadata = {
            "choices" : ["default", "2quad", "l2quad", "2relu"]
        }
    )
    training_phase: str = field(
        default="training",
        metadata = {
            "choices" : ["training", "analysis"]
        }
    )
    kl_method: str = field(
        default="forward",
        metadata = {
            "help" : "Possible options: 'forward'; 'reverse'; 'jsd{ratio}'"
        }
    )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    student_temperature: Optional[float] = field(default=1.0)
    teacher_temperature: Optional[float] = field(default=1.0)

def what_to_prune(
    head_importance,
    num_heads_to_prune,
    heads_per_layer,
    layers_to_prune,
    total_layers,
    to_prune=None,
    at_least_x_heads_per_layer=0,
):
    to_prune = to_prune or {}
    # Sort heads by score
    # only including scores for layers that are to be pruned
    heads_and_score = [
        ((layer, head), head_importance[idx, head])
        for idx, layer in enumerate(layers_to_prune)
        for head in range(heads_per_layer)
    ]
    heads_and_score = sorted(heads_and_score, key=lambda x: x[1])
    sorted_heads = [head_and_score[0]
                    for head_and_score in heads_and_score]
    # Ensure we don't delete all heads in a layer
    if at_least_x_heads_per_layer:
        # Remove the top scoring head in each layer
        to_protect = {l: 0 for l in range(total_layers)}
        filtered_sorted_heads = []
        for layer, head in reversed(sorted_heads):
            if layer in to_protect:
                if to_protect[layer] < at_least_x_heads_per_layer:
                    to_protect[layer] += 1
                    continue
                else:
                    to_protect.pop(layer)
            filtered_sorted_heads.insert(0, (layer, head))
        sorted_heads = filtered_sorted_heads
    # layer/heads that were already pruned
    # Prune the lowest scoring heads
    sorted_heads = [
        (layer, head)
        for (layer, head) in sorted_heads
        if layer not in to_prune or head not in to_prune[layer]
    ]
    # Update heads to prune
    for layer, head in sorted_heads[:num_heads_to_prune]:
        if layer not in to_prune:
            to_prune[layer] = set()
        to_prune[layer].add(head)
    # convert to list to make it JSON serializable
    for layer in to_prune:
        to_prune[layer] = list(to_prune[layer])
    return to_prune

def set_trainable_layers(model, trainable_list, use_lora):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for f in trainable_list:
            # if (f in name) and (use_lora => "lora" or "lm_head" or "norm" in name)
            if (f in name) and (not use_lora or ("lora" in name or "modules_to_save" in name)):
                param.requires_grad = True
    return model

def get_model_storage_path(model_path):
    from transformers.utils import default_cache_path
    import os
    try:
        # This checks if the path looks like a URL or is a local path that exists
        if not os.path.exists(model_path):
            # Model from the hub, find the cache location
            # `cached_path` can also directly show where remote resources are cached
            cache_directory = default_cache_path
            model_directory = os.path.join(cache_directory, model_path)
        else:
            # Local model path
            model_directory = os.path.abspath(model_path)

        return model_directory
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

class MarillConfig:
    def __init__(self, args: MarillTrainingArguments):
        self.layer = LayerConfig.from_str(args.layer_config)
        self.weight = WeightConfig.from_str(args.weight_config)
        self.head = HeadConfig.from_str(args.head_config)
        self.act = ActType.from_str(args.act_type)
        self.smax = SoftmaxType.from_str(args.smax_type)

    def default():
        return MarillConfig(MarillTrainingArguments(output_dir=""))

    def populate_layer_config(self, total_layers):
        layer_config = self.layer.as_dict()
        if self.layer in [LayerConfig.SpacedFreezing, LayerConfig.Pruning]:
            frozen_layers = self.layer.num_layers
            assert total_layers >= frozen_layers + 2
            num_trainable_layers = total_layers - frozen_layers
            layer_config["trainable_list"] = [ round((i / (num_trainable_layers - 1)) * (total_layers - 1)) for i in range(num_trainable_layers)]
        elif self.layer is LayerConfig.BottomFreezing:
            num_frozen_layers = self.layer.num_layers
            layer_config["trainable_list"] = list(range(num_frozen_layers, total_layers))
        elif self.layer is LayerConfig.Full:
            layer_config["trainable_list"] = list(range(total_layers))
        else:
            raise ValueError("Invalid LayerConfig")
        # useful in marill_config
        self.layer.trainable_list = layer_config["trainable_list"]
        return layer_config

    def populate_head_config(self, config, model_path):
        head_config = self.head.as_dict()
        if self.head in [HeadConfig.Pruning, HeadConfig.UniformPruning]:
            num_heads_to_prune = int(config.num_attention_heads * config.num_hidden_layers * (1.0 - 1.0/self.head.factor))
            try:
                if config.head_config["factor"] != self.head.factor:
                    sys.exit(1)
                to_prune = config.head_config["to_prune"]
            except Exception as e:
                abs_model_path = get_model_storage_path(model_path)
                head_importance_file = abs_model_path + "/head_importance.pt"
                head_config["head_importance_file"] = head_importance_file
                try:
                    head_importance = torch.load(head_importance_file, map_location=torch.device('cpu'))
                except FileNotFoundError:
                    assert(False), f"For head pruning, a head_importance.pt file must be present where model corresponding to `model_path` is stored: {head_importance_file}"
                # Only prune trained layers
                layers_to_prune = self.layer.trainable_list
                min_heads_per_layer = (1.0/self.head.factor) * config.num_attention_heads if self.head is HeadConfig.UniformPruning else 1
                to_prune_int_idx = what_to_prune(head_importance=head_importance, num_heads_to_prune=num_heads_to_prune, heads_per_layer=config.num_attention_heads, layers_to_prune=layers_to_prune, total_layers=config.num_hidden_layers, to_prune=None, at_least_x_heads_per_layer=min_heads_per_layer)
                to_prune = {}
                for layer in to_prune_int_idx:
                    to_prune[str(layer)] = to_prune_int_idx[layer]
                del to_prune_int_idx
            head_config["to_prune"] = to_prune
            rank0_print(f"Current Prune factor: {self.head.factor}; Pruning heads: {to_prune}")
            for layer in to_prune:
                rank0_print(f"Layer {layer}: {len(to_prune[layer])} heads")
        elif self.head in [HeadConfig.PermutedMerging, HeadConfig.Clustering, HeadConfig.EvenClustering]:
            num_heads_per_cluster = self.head.factor
            try:
                if config.head_config["factor"] != self.head.factor:
                    sys.exit(1)
                layerwise_clusters = config.head_config["clusters"]
                # if the given student model already has permuted merging, then we should not permute again and simply switch to merging
                if self.head is HeadConfig.PermutedMerging:
                    self.head = HeadConfig.Merging
            except Exception as e:
                abs_model_path = get_model_storage_path(model_path)
                head_similarity_file = abs_model_path + "/head_similarity.pt"
                head_config["head_similarity_file"] = head_similarity_file
                try:
                    head_similarity = torch.load(head_similarity_file, map_location=torch.device('cpu'))
                except FileNotFoundError:
                    assert(False), f"For permuted head merging or head clustering, a head_similarity.pt file must be present where model corresponding to `model_path` is stored: {head_similarity_file}"
                from marill_trainer.analyze_head_similarity import analyze_core as analyze_similarity
                clusters = analyze_similarity(similarity=head_similarity, num_clusters=config.num_attention_heads//num_heads_per_cluster, verbose=False)
                if self.head in [HeadConfig.PermutedMerging, HeadConfig.EvenClustering]:
                    clusters = clusters["even"]
                elif self.head is HeadConfig.Clustering:
                    clusters = clusters["uneven"]
                layerwise_clusters = {}
                for layer in range(config.num_hidden_layers):
                    layer_clusters = clusters[layer]
                    layerwise_clusters[str(layer)] = {}
                    for idx, cluster in enumerate(layer_clusters):
                        cluster_wo_scores = {"indices": cluster["indices"], "medoid_idx": cluster["medoid_idx"]}
                        layerwise_clusters[str(layer)][idx] = cluster_wo_scores
            head_config["clusters"] = layerwise_clusters
            self.head.clusters = layerwise_clusters

        return head_config

    def lora_and_permute_freeze_params(self, model):
        # permute the parameters if using permuted merging
        if self.head is HeadConfig.PermutedMerging:
            config = model.config
            assert config.num_attention_heads == config.num_key_value_heads
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
            head_dim = hidden_size // num_heads
            for layer_idx in self.head.clusters:
                layer = model.model.layers[int(layer_idx)]
                attn = layer.self_attn
                permutation = []
                clusters = self.head.clusters[layer_idx]
                for cluster_idx in clusters:
                    permutation.extend(clusters[cluster_idx]["indices"])
                assert(len(permutation) == config.num_attention_heads)
                # each weight is stored as (out_features, in_features)
                attn.q_proj.weight.data = attn.q_proj.weight.view(num_heads, head_dim, hidden_size)[permutation, :, :].view(num_heads * head_dim, hidden_size).contiguous()
                attn.k_proj.weight.data = attn.k_proj.weight.view(num_heads, head_dim, hidden_size)[permutation, :, :].view(num_heads * head_dim, hidden_size).contiguous()
                attn.v_proj.weight.data = attn.v_proj.weight.view(num_heads, head_dim, hidden_size)[permutation, :, :].view(num_heads * head_dim, hidden_size).contiguous()
                attn.o_proj.weight.data = attn.o_proj.weight.view(hidden_size, num_heads, head_dim)[:, permutation, :].view(hidden_size, num_heads * head_dim).contiguous()

        # if base_model is already wrapped in lora, we shouldn't wrap it again
        if self.weight is WeightConfig.LoRA and not isinstance(model, PeftModel):
            '''
            It is okay to simply set requires_grad=False for LoRA weights in frozen layers 
            The reason is that at least one of the low-rank matrix is initialized with zeros. So the product of the matrices is zero and shouldn't affect the output.
            This is atleast true in the PEFT implementation we're using (v0.4.0): See https://github.com/huggingface/peft/blob/0769587a3cd80ad2ae508cc06efbf54ddca821b3/src/peft/tuners/lora.py#L738
            '''
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"], r=self.weight.rank, lora_alpha=32, lora_dropout=0.1, init_lora_weights=True, layers_to_transform=self.layer.trainable_list, modules_to_save=["embed_tokens", "lm_head"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        if self.layer in [LayerConfig.BottomFreezing, LayerConfig.SpacedFreezing, LayerConfig.Pruning]:
            trainable_list = ["layers." + str(i) + "." for i in self.layer.trainable_list]
            trainable_list += ["lm_head"]
            if self.layer.num_layers == 0 or self.layer is LayerConfig.Pruning:
                trainable_list += ["embed_tokens"]
            set_trainable_layers(model, trainable_list, self.weight is WeightConfig.LoRA)

        for name, param in model.named_parameters():
            if param.requires_grad is False:
                rank0_print(f"Non-Trainable: {name}")
            else:
                rank0_print(f"Trainable: {name}")
        total_params = sum([p.numel() for p in model.parameters()])
        total_trainable_params = sum(
            [p.numel() if p.requires_grad else 0 for p in model.parameters()]
        )
        rank0_print(
            f"Total number of params in student: {total_params}, trainable: {total_trainable_params}, fraction: {total_trainable_params/total_params}"
        )

        return model

    def __str__(self) -> str:
        ret = ""
        if self.layer is not LayerConfig.Full:
            ret += f"_{self.layer}"
        if self.weight is not WeightConfig.Full:
            ret += f"_{self.weight}"
        if self.head is not HeadConfig.Default:
            ret += f"_{self.head}"
        if self.act is not ActType.Default:
            ret += f"_{self.act}"
        if self.smax is not SoftmaxType.Default:
            ret += f"_{self.smax}"
        if ret == "":
            ret = "default"
        return ret

'''
TODO:
~set common config params: hidden_act, softmax_act, layer_config, head_config, skip_flash_attn, calculate_head_importance
~set layer-wise config params: head_config.head_mask[num_layers]
~save pruning analysis in the same directory as the model being analysed; abort if not found during pruning
~setup lora
~setup layer freezing
~wandb init
~load the student_model for the first time within the trainer init
for cases with multi-step training, ensure that the model's config and cmdline config match for head_merging, softmax, activation, etc.
for the pre-trained model, save the analysis in the output directory
~infer marill configuration from name
~don't load teacher if no distillation
~training phases: fine-tuning and analysis
~only perform distillation for layers which were trained
'''

def hash_command_line_arguments():
    # Extract command line arguments excluding the script name
    arguments = sys.argv[1:]
    rank0_print(arguments)
    # Concatenate all command line arguments into a single string
    concatenated_arguments = ' '.join(arguments)
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the concatenated arguments
    sha256_hash.update(concatenated_arguments.encode('utf-8'))
    # Get the hexadecimal representation of the hash digest
    hashed_arguments = sha256_hash.hexdigest()
    return hashed_arguments[:6]

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    torch.distributed.barrier()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()
        if trainer.marill_config.weight is WeightConfig.LoRA:
            from peft import get_peft_model_state_dict
            state_dict = get_peft_model_state_dict(trainer.model)
            for name, tensor in state_dict.items():
                rank0_print(f"Saving {name} with shape {tensor.shape}")
            import os
            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, os.path.join(trainer.args.output_dir, "adapter_model.bin"))
            # PEFT save_pretrained is buggy
            # trainer.model.save_pretrained(trainer.args.output_dir, safe_serialization=False, is_main_process=torch.distributed.get_rank() == 0)
    torch.distributed.barrier()

class MarillTrainer(Trainer):
    def __init__(
        self,
        model_path: str,
        teacher_model_path: str,
        args: MarillTrainingArguments,
        project_name: str,
        tokenizer: transformers.PreTrainedTokenizer,
        *aargs,
        **kwargs,
    ):
        torch.distributed.barrier()
        self.tokenizer = tokenizer
        self.phase = TrainingPhase.from_str(args.training_phase)
        self.model_path = model_path
        self.kl_method = KLMethod.from_str(args.kl_method)
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            cache_dir=args.cache_dir,
            use_cache=False
        )
        marill_config = MarillConfig(args)
        save_and_log_name = f"{marill_config}_{hash_command_line_arguments()}"

        if self.phase is TrainingPhase.HEAD_ANALYSIS:
            rank0_print("Not applying any MARILL config changes during analysis and skipping flash_attn")
            marill_config = MarillConfig.default()
            config.skip_flash_attn = True

        self.marill_config = marill_config
        config = MarillTrainer.process_config(config, model_path, marill_config, self.phase)
        torch.distributed.barrier()
        rank0_print(config)
        self.processed_config = config
        args.output_dir = args.output_dir + f"_{save_and_log_name}"
        if torch.distributed.get_rank() == 0 and self.phase is not TrainingPhase.HEAD_ANALYSIS:
            run = wandb.init(reinit=True, project=project_name, name=save_and_log_name, config=args)
        
        # loading LoRA model is slightly different
        base_model_uses_lora = "lora" in model_path
        if base_model_uses_lora:
            rank0_print("Base Model uses LoRA.")
            peft_config_base = PeftConfig.from_pretrained(model_path)
            base_model = LlamaForCausalLMStudent.from_pretrained(peft_config_base.base_model_name_or_path, config=config)
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = LlamaForCausalLMStudent.from_pretrained(
                model_path,
                config=config,
            )

        # permute/freeze params and/or apply LoRA according to marill config
        model = marill_config.lora_and_permute_freeze_params(model)
        rank0_print(f"student model: {model}")

        self.using_distillation = (args.logit_coef > 0.0 or args.hs_coef > 0.0 or args.att_coef > 0.0)
        self.using_att_loss = args.att_coef > 0.0
        self.using_hs_loss = args.hs_coef > 0.0
        if (self.phase is not TrainingPhase.HEAD_ANALYSIS) and (self.using_distillation):
            other_kwargs: dict = {}
            if transformers_version >= "4.35.0":
                # enable flash_attn_2 to pass the right attention_mask when evaluating teacher
                other_kwargs["use_flash_attention_2"] = True
            teacher_config = transformers.AutoConfig.from_pretrained(
                teacher_model_path,
                cache_dir=args.cache_dir,
                use_cache=False,
            )
            teacher_model = LlamaForCausalLMTeacher.from_pretrained(
                teacher_model_path,
                config=teacher_config,
                **other_kwargs
            )
            rank0_print("teacher_config", teacher_config)
            self.teacher_model = teacher_model.to(device="cuda", dtype=torch.bfloat16)
            if self.using_att_loss or self.using_hs_loss:
                args.gradient_checkpointing = False
            else:
                args.gradient_checkpointing = True
        else:
            self.teacher_model = None
            args.gradient_checkpointing = True

        if self.phase is TrainingPhase.HEAD_ANALYSIS:
            self.n_layers = config.num_hidden_layers
            self.n_heads = config.num_attention_heads
            self.layers_to_prune = marill_config.layer.trainable_list
            self.head_importance = torch.zeros(len(self.layers_to_prune), self.n_heads).cuda()
            self.head_similarity = torch.zeros(self.n_layers, self.n_heads, self.n_heads).cuda()
            self.tot_tokens_importance = 0
            self.tot_tokens_similarity = 0
            self.normalize_scores_by_layer = True
            args.gradient_checkpointing = False

        kwargs['model'] = model
        kwargs['args'] = args
        kwargs['tokenizer'] = tokenizer
        super().__init__(*aargs, **kwargs)

        self.att_loss = torch.tensor(0.0).to("cuda")
        self.hs_loss = torch.tensor(0.0).to("cuda")
        self.logits_loss = torch.tensor(0.0).to("cuda")
        self.label_loss = torch.tensor(0.0).to("cuda")
        self.label_loss_teacher = torch.tensor(0.0).to("cuda")

        self.total_att_loss = 0.0
        self.total_hs_loss = 0.0
        self.total_logits_loss = 0.0
        self.total_label_loss = 0.0
        self.total_label_loss_teacher = 0.0

        self.step_last_logged = self.state.global_step

    def finalize(self):
        if self.phase is TrainingPhase.TRAIN:
            torch.distributed.barrier()
            self.model.config.use_cache = True
            self.save_state()
            trainer_save_model_safe(self)
            if self.marill_config.weight is WeightConfig.LoRA:
                # when using LoRA, also store the config file for the BaseModel
                self.processed_config.save_pretrained(self.args.output_dir)
                '''
                # make sure adapter weights are saved
                self.model.save_pretrained(self.args.output_dir)
                torch.distributed.barrier()
                '''
            # self.tokenizer.save_pretrained(self.args.output_dir)
        elif self.phase is TrainingPhase.HEAD_ANALYSIS:
            head_importance = self.get_head_importance()
            head_similarity = self.get_head_similarity()
            abs_model_path = get_model_storage_path(self.model_path)
            head_importance_file = abs_model_path + "/head_importance.pt"
            head_similarity_file = abs_model_path + "/head_similarity.pt"
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                torch.save(head_importance, head_importance_file)
                torch.save(head_similarity, head_similarity_file)
        else:
            raise ValueError(f"Invalid TrainingPhase: {self.phase}")

    def get_marill_eval_model(model_path):
        marill_config = transformers.AutoConfig.from_pretrained(
            model_path,
            use_cache=True
        )
        # skip flash_attn when evaluating the model
        marill_config.skip_flash_attn = True
        print("marill_config", marill_config)
        if "lora" in model_path:
            rank0_print("Base Model uses LoRA.")
            peft_config_base = PeftConfig.from_pretrained(model_path)
            base_model = LlamaForCausalLMStudent.from_pretrained(peft_config_base.base_model_name_or_path, config=marill_config)
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            model = LlamaForCausalLMStudent.from_pretrained(
                model_path,
                config=marill_config,
            )
        rank0_print("eval model", model)
        return model

    def process_config(config, model_path, marill_config, phase):
        # set common config params: hidden_act, softmax_act, layer_config, head_config, skip_flash_attn, calculate_head_importance
        # set layer-wise config params: head_config.head_mask[num_layers]
        config.hidden_act = str(marill_config.act)
        config.softmax_act = str(marill_config.smax)

        config.layer_config = marill_config.populate_layer_config(config.num_hidden_layers)
        config.head_config = marill_config.populate_head_config(config, model_path)

        if phase is TrainingPhase.HEAD_ANALYSIS:
            config.head_analysis = True
        else:
            config.head_analysis = False

        if (config.softmax_act not in ["default", "smax"]) or phase is TrainingPhase.HEAD_ANALYSIS:
            config.skip_flash_attn = True
        else:
            config.skip_flash_attn = False

        return config

    def get_head_importance(self):
        importance = self.head_importance / self.tot_tokens_importance

        # gather the importance scores from all the processes
        # after gather, the shape is [n_gpu * n_layers, n_heads]
        importance = self._nested_gather(importance).view(-1, len(self.layers_to_prune), self.n_heads).mean(0)

        # Layerwise importance normalization
        if self.normalize_scores_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(importance, exponent).sum(-1), 1/exponent)
            importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        return importance

    def get_head_similarity(self):
        similarity = self.head_similarity / self.tot_tokens_similarity

        # gather the importance scores from all the processes
        # after gather, the shape is [n_gpu * n_layers, n_heads, n_heads]
        similarity = self._nested_gather(similarity).view(-1, self.n_layers, self.n_heads, self.n_heads).mean(0)

        # Layerwise importance normalization
        '''
        if self.normalize_scores_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(importance, exponent).sum(-1), 1/exponent)
            importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        '''
        return similarity

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        if self.is_in_train == True:
            # all_gather + mean() to get average loss over all processes
            att_loss_scalar = self._nested_gather(self.att_loss).mean().item()
            hs_loss_scalar = self._nested_gather(self.hs_loss).mean().item()
            logits_loss_scalar = self._nested_gather(self.logits_loss).mean().item()
            label_loss_scalar = self._nested_gather(self.label_loss).mean().item()
            label_loss_teacher_scalar = self._nested_gather(self.label_loss_teacher).mean().item()
            # reset loss to zero
            self.att_loss -= self.att_loss
            self.hs_loss -= self.hs_loss
            self.logits_loss -= self.logits_loss
            self.label_loss -= self.label_loss
            self.label_loss_teacher -= self.label_loss_teacher
            num_steps = self.state.global_step - self.step_last_logged
            # update logs
            logs["att_loss"] = round(att_loss_scalar / num_steps, 4)
            logs["hs_loss"] = round(hs_loss_scalar / num_steps, 4)
            logs["logits_loss"] = round(logits_loss_scalar / num_steps, 4)
            logs["label_loss"] = round(label_loss_scalar / num_steps, 4)
            logs["label_loss_teacher"] = round(label_loss_teacher_scalar / num_steps, 4)
            # update total loss
            self.total_att_loss += att_loss_scalar
            self.total_hs_loss += hs_loss_scalar
            self.total_logits_loss += logits_loss_scalar
            self.total_label_loss += label_loss_scalar
            self.total_label_loss_teacher += label_loss_teacher_scalar
            # update step_last_logged
            self.step_last_logged = self.state.global_step
        else:
            # add remaining loss to total loss
            self.total_att_loss += self.att_loss.item()
            self.total_hs_loss += self.hs_loss.item()
            self.total_logits_loss += self.logits_loss.item()
            self.total_label_loss += self.label_loss.item()
            self.total_label_loss_teacher += self.label_loss_teacher.item()
            # update final log 
            logs["att_loss"] = self.total_att_loss / self.state.global_step
            logs["hs_loss"] = self.total_hs_loss / self.state.global_step
            logs["logits_loss"] = self.total_logits_loss / self.state.global_step
            logs["label_loss"] = self.total_label_loss / self.state.global_step
            logs["label_loss_teacher"] = self.total_label_loss_teacher / self.state.global_step

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    # changing the loss function to include loss from attentions and hidden states
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.phase is TrainingPhase.HEAD_ANALYSIS:
            outputs = model(**inputs)
            loss = outputs.get("loss")
            loss.mean()
            # need to call backward on loss to get gradients
            self.accelerator.backward(loss)
            attention_mask = inputs["attention_mask"]
            for idx, layer in enumerate(self.layers_to_prune):
                self_attn = model.model.layers[layer].self_attn
                # ctx shape: (bsz, q_len, num_heads, head_dim)
                ctx = self_attn.context_layer_val
                grad_ctx = ctx.grad
                # Take the dot
                dot = torch.einsum("bqhd,bqhd->bhq", [grad_ctx, ctx])
                importance = dot.abs().sum(-1).sum(0).detach()
                # importance = self._nested_gather(importance).view(-1, self.n_heads)
                self.head_importance[idx] += importance
                del ctx, grad_ctx
            self.tot_tokens_importance += attention_mask.float().detach().sum().data

            for layer in range(self.n_layers):
                attn_weights = model.model.layers[layer].self_attn.attn_weights.detach()
                # attn_weights shape: (bsz, num_heads, q_len, kv_seq_len)
                for head1 in range(self.n_heads):
                    for head2 in range(head1 + 1, self.n_heads):
                        similarity = jsd(attn_weights[:, head1], attn_weights[:, head2])
                        # similarity = torch.nn.functional.cosine_similarity(attn_weights[:, head1], attn_weights[:, head2], dim=-1).sum(0)
                        self.head_similarity[layer, head1, head2] += similarity
                        self.head_similarity[layer, head2, head1] = self.head_similarity[layer, head1, head2]
                del attn_weights
            self.tot_tokens_similarity += 1

            # set the gradients to 0 so that the model is not updated
            model.zero_grad()
            # Just calculating the importance here; putting a dummy loss here that does not update the model and does not throw an error when backward is invoked by Huggingface trainer
            loss = outputs.loss.new_tensor(0.0)
            loss.requires_grad = True
            loss.sum()
            return (loss, outputs) if return_outputs else loss

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # output_mask = inputs["labels"][..., 1:] == IGNORE_TOKEN_ID
        output_mask = inputs["labels"][..., :] == IGNORE_TOKEN_ID
        labels = inputs["labels"]

        #################################################
        ######## LOSS COMPUTATION LOGIC ################# 
        #################################################

        self.step_last_logged = self._globalstep_last_logged
        num_steps = (1 + self.state.global_step - self.step_last_logged)

        outputs_student = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=self.using_att_loss,
            output_hidden_states=self.using_hs_loss,
            use_cache=False,
        )
        label_loss = outputs_student.get("loss")
        logits_student = outputs_student.get("logits")
        loss = self.args.label_coef * label_loss

        # forward pass
        if self.using_distillation:
            logits_loss = 0.0
            att_loss = 0.0
            hs_loss = 0.0

            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=self.using_att_loss,
                    output_hidden_states=self.using_hs_loss,
                    use_cache=False,
                )
            label_loss_teacher = outputs_teacher.get("loss")
            logits_teacher = outputs_teacher.get("logits")
            loss_mse = MSELoss()

            if self.using_hs_loss:
                # the first hidden_state is the input embeddings
                hss_student = outputs_student.get("hidden_states")[1:]
                hss_teacher = outputs_teacher.get("hidden_states")[-len(hss_student) :]
                for hs_student, hs_teacher in zip(hss_student, hss_teacher):
                    if hs_student is None:
                        continue
                    hs_teacher = hs_teacher.type(hs_student.dtype)
                    tmp_loss = loss_mse(hs_student, hs_teacher)
                    hs_loss += tmp_loss
                    del hs_student, hs_teacher
                del hss_student, hss_teacher
                loss += self.args.hs_coef * hs_loss
                hs_loss_step = (hs_loss.mean().detach() / self.args.gradient_accumulation_steps)
                if (self.args.logging_nan_inf_filter and (torch.isnan(hs_loss_step) or torch.isinf(hs_loss_step))):
                    self.hs_loss += self.hs_loss / num_steps
                else:
                    self.hs_loss += hs_loss_step

            if self.using_att_loss:
                atts_student = outputs_student.get("attentions")
                # ignore the values in the teacher output that are not in the student output
                atts_teacher = outputs_teacher.get("attentions")[-len(atts_student) :]
                # if output_context = True, the output of self_attention is returned with output_attentions=True
                for att_student, att_teacher in zip(atts_student, atts_teacher):
                    if att_student is None:
                        continue
                    att_teacher = att_teacher.type(att_student.dtype)
                    tmp_loss = loss_mse(att_student, att_teacher)
                    att_loss += tmp_loss
                    del att_student, att_teacher
                del atts_teacher, atts_student
                loss += self.args.att_coef * att_loss
                att_loss_step = (att_loss.mean().detach() / self.args.gradient_accumulation_steps)
                if (self.args.logging_nan_inf_filter and (torch.isnan(att_loss_step) or torch.isinf(att_loss_step))):
                    self.att_loss += self.att_loss / num_steps
                else:
                    self.att_loss += att_loss_step
                #'''

            logits_teacher = logits_teacher.type(logits_student.dtype)
            if self.kl_method == KLMethod.Forward:
                logits_loss = self.soft_cross_entropy(
                    logits_student / self.args.student_temperature,
                    logits_teacher / self.args.teacher_temperature,
                    output_mask
                )
            elif self.kl_method == KLMethod.Reverse:
                logits_loss = self.get_kl(
                    logits_teacher / self.args.teacher_temperature,
                    logits_student / self.args.student_temperature,
                    output_mask
                )
            elif self.kl_method == KLMethod.JSD:
                assert self.args.student_temperature == 1.0
                assert self.args.teacher_temperature == 1.0
                fwd_loss_ratio = self.kl_method.ratio
                logits_interpolate = fwd_loss_ratio * logits_student + (1 - fwd_loss_ratio) * logits_teacher
                fwd_loss = self.get_kl(
                    logits_student,
                    logits_interpolate,
                    output_mask
                )
                reverse_loss = self.get_kl(
                    logits_teacher,
                    logits_interpolate,
                    output_mask
                )
                logits_loss = fwd_loss_ratio * fwd_loss + (1 - fwd_loss_ratio) * reverse_loss
            loss += self.args.logit_coef * logits_loss
            logits_loss_step = (logits_loss.mean().detach() / self.args.gradient_accumulation_steps)
            label_loss_teacher_step = (label_loss_teacher.mean().detach() / self.args.gradient_accumulation_steps)
            if (self.args.logging_nan_inf_filter and (torch.isnan(logits_loss_step) or torch.isinf(logits_loss_step))):
                self.logits_loss += self.logits_loss / num_steps
                print(f"NaN output from student (rank {dist.get_rank()})")
            else:
                self.logits_loss += logits_loss_step
            if (self.args.logging_nan_inf_filter and (torch.isnan(label_loss_teacher_step) or torch.isinf(label_loss_teacher_step))):
                self.label_loss_teacher += self.label_loss_teacher / num_steps
                print(f"NaN output from teacher (rank {dist.get_rank()})")
            else:
                self.label_loss_teacher += label_loss_teacher_step

        label_loss_step = (label_loss.mean().detach() / self.args.gradient_accumulation_steps)
        # if loss is nan or inf simply add the average of previous logged losses
        if (self.args.logging_nan_inf_filter and (torch.isnan(label_loss_step) or torch.isinf(label_loss_step))):
            self.label_loss += self.label_loss / num_steps
        else:
            self.label_loss += label_loss_step

        return loss

    ###################### Helper Functions #############################
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_kl(self, predicts, targets, padding_mask):
        kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        predict_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.log_softmax(targets, dim=-1)
        output = kl_loss(predict_prob, targets_prob)
        expand_mask = padding_mask.unsqueeze(-1).expand_as(output)
        output.masked_fill_(expand_mask, 0)
        mean_output = output.sum() / (~padding_mask).sum()
        return mean_output

def kld(a1, a2, smoothing=1e-10) :
    log_a1 = (a1 + smoothing).log()
    log_a2 = (a2 + smoothing).log()
    kld_v = a1 * (log_a1 - log_a2)
    return kld_v.sum() / a1.size(0)

def jsd(p, q):
    # rank0_print("p shape", p.shape)
    # rank0_print("q shape", q.shape)
    # torch.distributed.barrier()
    p, q = p.reshape(-1, p.size(-1)), q.reshape(-1, q.size(-1))
    m = 0.5 * (p + q)
    jsd_v = 0.5 * (kld(p, m) + kld(q, m))
    return jsd_v