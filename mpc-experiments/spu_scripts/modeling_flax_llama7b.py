import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import tempfile
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

from einops import rearrange
import functools
from typing import NamedTuple

USE_LORA = False
RANK = 4

def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]

"""
Computing ffn blockwise without materializing the large hidden tensor, training
4x longer sequences than the memory-efficient transformer.
Blockwise parallel transformer https://arxiv.org/abs/2305.19370 Liu et al. 2023
"""
def blockwise_ffn(remat_ffn, inputs, chunk_size=2048, deterministic=True):
    # remat_ffn: a rematerialized ffn with policy jax.checkpoint_policies.nothing_saveable()
    # inputs: (batch, seq_len, dim)
    # chunk_size: the chunk size to split the sequence
    inputs = rearrange(inputs, 'b (c n) d -> b c n d', c=chunk_size)
    def scan_ffn(remat_ffn, carry, hidden_states):
        outputs = remat_ffn(hidden_states, deterministic=deterministic)
        return carry, outputs
    scan_axis = inputs.ndim - 2
    _, res = nn.scan(
        scan_ffn,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=scan_axis,
        out_axes=scan_axis,
    )(remat_ffn, None, inputs)
    res = rearrange(res, 'b c n d -> b (c n) d')
    return res

class Carry(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    max_so_far: jax.Array

def _chunk_attention_bias(query_chunk_size, key_chunk_size,
            bias, deterministic, attn_dropout, attn_pdrop, causal,
            dtype, query_chunk_idx, key_chunk_idx):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )

    if causal:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)

"""
Compute attention blockwise without materializing the full attention matrix,
initially proposed in memory-efficient transformer https://arxiv.org/abs/2112.05682 Rabe et al. 2021;
flash attention https://arxiv.org/abs/2205.14135 Dao et al. 2022 proposes a CUDA
efficient implementation; blockwise parallel transformer https://arxiv.org/abs/2305.19370
Liu et al. 2023 proposes blockwise computing both attention and FFN, enabling 4x
longer sequences than memory-efficient/flash-attention and fusion of attention and FFN.
"""
def blockwise_attn(
        query, key, value,
        bias=None,
        deterministic=True,
        dropout_rng=None,
        attn_pdrop=0.0,
        causal=True,
        query_chunk_size=2048,
        key_chunk_size=2048,
        dtype=jnp.float32,
        policy=jax.checkpoint_policies.nothing_saveable(),
        precision=None,
        float32_logits=True,
        prevent_cse=True,
    ):
    # query, key, value: (batch, seq_len, num_heads, dim_per_head)
    # bias: (batch, seq_len) can be used to mask out attention (e.g. padding)
    # causal: whether to use causal mask
    # policy: one of jax.checkpoint_policies
    query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape

    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))

    query = jnp.moveaxis(query, 1, 0)
    key = jnp.moveaxis(key, 1, 0)
    value = jnp.moveaxis(value, 1, 0)

    if bias is not None:
        for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
            assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = functools.partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)

    def scan_attention(args):
        query_chunk, query_chunk_idx = args

        @functools.partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk, precision=precision)
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = jnp.einsum(
                'bqhv,bvhd->bqhd', exp_weights, value_chunk, precision=precision
            )
            correction = jnp.exp(prev_max_score - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)
            return Carry(numerator, denominator, max_score), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = query_chunk_idx < key_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args,
            )

        init_carry = Carry(
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=query.dtype),
        )
        (numerator, denominator, max_score), _ = lax.scan(
            skip_upper_half, init_carry, xs=(key, value, jnp.arange(0, num_kv))
        )
        outputs = (numerator / denominator).astype(dtype)
        return outputs

    _, res = lax.scan(
        lambda _, x: ((), scan_attention(x)),
        (), xs=(query, jnp.arange(0, num_q))
    )
    res = rearrange(res, 'n b c h d -> b (n c) h d')
    return res

LLAMA_STANDARD_CONFIGS = {
    '1b': {
        'vocab_size': 32000,
        'hidden_size': 2048,
        'intermediate_size': 5504,
        'num_hidden_layers': 22,
        'num_attention_heads': 16,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '3b': {
        'vocab_size': 32000,
        'hidden_size': 3200,
        'intermediate_size': 8640,
        'num_hidden_layers': 26,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '7b': {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '13b': {
        'vocab_size': 32000,
        'hidden_size': 5120,
        'intermediate_size': 13824,
        'num_hidden_layers': 40,
        'num_attention_heads': 40,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '30b': {
        'vocab_size': 32000,
        'hidden_size': 6656,
        'intermediate_size': 17920,
        'num_hidden_layers': 60,
        'num_attention_heads': 52,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '65b': {
        'vocab_size': 32000,
        'hidden_size': 8192,
        'intermediate_size': 22016,
        'num_hidden_layers': 80,
        'num_attention_heads': 64,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    'debug': { # A small model for debugging
        'vocab_size': 32000,
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
}


class LLaMAConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_sequence_length=2048,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        # pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        remat_block='',
        remat_attention='',
        remat_mlp='',
        scan_attention=False,
        scan_mlp=False,
        scan_query_chunk_size=1024,
        scan_key_chunk_size=1024,
        scan_mlp_chunk_size=1024,
        fcm_min_ratio=0.0,
        fcm_max_ratio=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.remat_block = remat_block
        self.remat_attention = remat_attention
        self.remat_mlp = remat_mlp
        self.scan_attention = scan_attention
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        super().__init__(
            # pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

remat = nn_partitioning.remat

logger = logging.get_logger(__name__)

class RMSNorm(nn.Module):
    dim: int
    eps: float=1e-6
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


class FlaxLLaMAAttention(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        config = self.config
        if config.head_pruning is not None:
            self.embed_dim = config.pruned_hidden_size
        else:
            self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if USE_LORA:
            self.wq_A = nn.Dense(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wq_B = nn.Dense(
                config.num_attention_heads*self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wk_A = nn.Dense(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wk_B = nn.Dense(
                config.num_attention_heads*self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wv_A = nn.Dense(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wv_B = nn.Dense(
                config.num_attention_heads*self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wo_A = nn.Dense(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wo_B = nn.Dense(
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
        else:
            self.wq = nn.Dense(
                config.num_attention_heads*self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wk = nn.Dense(
                config.num_attention_heads*self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wv = nn.Dense(
                config.num_attention_heads*self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.wo = nn.Dense(
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool")

        '''
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.max_sequence_length * 2,
            dtype=self.dtype,
        )
        #'''

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
    ):
        if USE_LORA:
            xq, xk, xv = self.wq_B(self.wq_A(hidden_states)), self.wk_B(self.wk_A(hidden_states)), self.wv_B(self.wv_A(hidden_states))
        else:
            xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        # xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
        # xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
        # xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        ''' skip, anyway it's a public-private multiplication
        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)
        #'''

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.config.scan_attention and not (self.has_variable("cache", "cached_key") or init_cache):
            # doesn't need blockwise attention if we are doing autoregressive decoding since no quadratic memory

            # attention mask without nxn materlization, blockwise_attn will handle the rest
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            # transform boolean mask into float mask
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_weights = None
            attn_output = blockwise_attn(
                xq,
                xk,
                xv,
                bias=attention_bias,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
                attn_pdrop=self.config.attn_pdrop,
                causal=True,
                query_chunk_size=self.config.scan_query_chunk_size,
                key_chunk_size=self.config.scan_key_chunk_size,
                dtype=self.dtype,
                policy=get_gradient_checkpoint_policy('nothing_saveable'),
                precision=self.precision,
                float32_logits=True,
                prevent_cse=True,
            )
            # attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp", None))
        else:
            query_length, key_length = xq.shape[1], xk.shape[1]

            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]

            batch_size = hidden_states.shape[0]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

            # transform boolean mask into float mask
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_weights = dot_product_attention_weights(
                xq,
                xk,
                bias=attention_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attn_pdrop,
                deterministic=deterministic,
                dtype=jnp.promote_types(self.dtype, jnp.float32),
                precision=self.precision,
            )
            # attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)

        attn_output = self._merge_heads(attn_output)
        if USE_LORA:
            attn_output = self.wo_B(self.wo_A(attn_output))
        else:
            attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxLLaMAMLP(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        config = self.config

        if USE_LORA:
            self.w1_A = nn.Dense(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.w1_B = nn.Dense(
                config.intermediate_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.w2_A = nn.Dense(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.w2_B = nn.Dense(
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.w3_A = nn.Dense(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.w3_B = nn.Dense(
                config.intermediate_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
        else:
            self.w1 = nn.Dense(
                config.intermediate_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.w2 = nn.Dense(
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
            self.w3 = nn.Dense(
                config.intermediate_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                precision=self.precision,
            )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if USE_LORA:
            x = self.w2_B(self.w2_A(nn.silu(self.w1_B(self.w1_A(x))) * self.w3_B(self.w3_A(x))))
        else:
            x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x


class FlaxLLaMABlock(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        attention_module = FlaxLLaMAAttention
        mlp_module = FlaxLLaMAMLP
        if self.config.remat_attention != '':
            attention_module = remat(
                FlaxLLaMAAttention, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention),
                prevent_cse=True,
            )
        if self.config.remat_mlp != '':
            mlp_module = remat(
                FlaxLLaMAMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp),
                prevent_cse=True,
            )

        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask: Optional[jnp.ndarray] = None,
    ):
        attn_outputs = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ffn_norm(hidden_states)

        if self.config.scan_mlp:
            feed_forward_hidden_states = blockwise_ffn(
                self.feed_forward,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
                deterministic,
            )
        else:
            feed_forward_hidden_states = self.feed_forward(
                feed_forward_input,
                deterministic,
            )
        # feed_forward_hidden_states = with_sharding_constraint(feed_forward_hidden_states, PS(("dp", "fsdp"), None, "mp"))

        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]


class FlaxLLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LLaMAConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: LLaMAConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxLLaMABlockCollection(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        block = FlaxLLaMABlock
        if self.config.remat_block != '':
            block = remat(
                FlaxLLaMABlock, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        self.blocks = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            ) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if not deterministic and self.config.fcm_max_ratio > 0:
            # Apply forgetful causal mask
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            fcm_ratio = jax.random.uniform(
                self.make_rng('fcm'), shape=(batch_size, 1, 1, 1),
                minval=self.config.fcm_min_ratio,
                maxval=self.config.fcm_max_ratio
            )
            fcm_mask = jax.random.uniform(
                self.make_rng('fcm'),
                shape=(batch_size, 1, 1, seq_length)
            ) > fcm_ratio
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype('bool')
        else:
            fcm_mask = None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                fcm_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLLaMAModule(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxLLaMABlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.ln_f = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )

@add_start_docstrings("", "")
class FlaxLLaMAModel(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAModule

# append_call_sample_docstring(
#     FlaxLLaMAModel,
#     _TOKENIZER_FOR_DOC,
#     _CHECKPOINT_FOR_DOC,
#     FlaxCausalLMOutput,
#     _CONFIG_FOR_DOC,
# )

class FlaxLLaMAForCausalLMModule(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.transformer = FlaxLLaMAModule(self.config, dtype=self.dtype)
        if self.config.skip_lm:
            print("using HackDense")
            from jax_utils import HackDense
            DenseModule = HackDense
        else:
            DenseModule = nn.Dense
        if USE_LORA:
            self.lm_head_A = DenseModule(
                RANK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
                precision=self.precision,
            )
            self.lm_head_B = DenseModule(
                self.config.vocab_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
                precision=self.precision,
            )
        else:
            self.lm_head = DenseModule(
                self.config.vocab_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
                precision=self.precision,
            )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        assert(self.config.tie_word_embeddings == False)
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            if USE_LORA:
                lm_logits = self.lm_head_B(self.lm_head_A(hidden_states))
            else:
                lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings("", "")
class FlaxLLaMAForCausalLM(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPTJ uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs