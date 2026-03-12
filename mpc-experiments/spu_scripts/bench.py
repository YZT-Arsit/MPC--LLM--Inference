# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_gpt2/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_gpt2:flax_gpt2

import argparse
import json

import jax.numpy as jnp
from transformers import AutoTokenizer
import transformers
import jax
import flax
from flax.linen.linear import Array
from contextlib import contextmanager
from typing import Optional, Tuple, Union
import jax.nn as jnn
import flax.linen as nn
import spu.utils.distributed as ppd

import time, json
import spu.utils.simulation as pps
import spu

from transformers.generation.flax_utils import GreedyState, FlaxGreedySearchOutput
from transformers.generation.flax_logits_process import FlaxLogitsProcessorList
from typing import Dict, Any, Callable

copts = spu.spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

#############################################################################
# Switching greedy_search to make the while condition independent of secrets
#############################################################################

def _greedy_search(
    self,
    input_ids: None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    logits_processor: Optional[FlaxLogitsProcessorList] = None,
    trace: bool = True,
    params: Optional[Dict[str, jnp.ndarray]] = None,
    model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    state: Optional[GreedyState] = None,
):
    decoding_only = state is not None

    # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
    # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
    model = self.decode if self.config.is_encoder_decoder else self

    if not decoding_only:
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = jax.lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = GreedyState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

    def greedy_search_cond_fn(state):
        """state termination condition fn."""
        has_reached_max_length = state.cur_len == max_length
        # all_sequence_finished = jnp.all(state.is_sent_finished)
        # finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
        finish_generation = has_reached_max_length
        return ~finish_generation

    def greedy_search_body_fn(state):
        """state update fn."""
        model_outputs = model(state.running_token, params=params, **state.model_kwargs)
        logits = model_outputs.logits[:, -1]

        # apply min_length, ...
        # logits = logits_processor(state.sequences, logits, state.cur_len)

        next_token = jnp.argmax(logits, axis=-1)

        # next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        next_is_sent_finished = state.is_sent_finished# | (next_token == eos_token_id)
        next_token = next_token[:, None]

        next_sequences = jax.lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
        next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
        return GreedyState(
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sent_finished=next_is_sent_finished,
            model_kwargs=next_model_kwargs,
        )

    if decoding_only:
        # perform the remaining decoding steps
        state = jax.lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)
        return FlaxGreedySearchOutput(sequences=state.sequences)
    else:
        # only perform the prefilling and the first iteration
        state = greedy_search_body_fn(state)
        return state, max_length

    '''
    # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
    if input_ids.shape[1] > 1:
        state = greedy_search_body_fn(state)

    if not trace:
        state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
    else:

    return FlaxGreedySearchOutput(sequences=state.sequences)
    '''

transformers.generation.FlaxGenerationMixin._greedy_search = _greedy_search
print("Switched `transformers.generation.FlaxGenerationMixin._greedy_search` to our custom `_greedy_search`")

#############################################################################
# Hacking softmax, gelu and silu with their approximations from PUMA
#############################################################################

import enum
class GeLUType(enum.Enum):
    Default = 0
    Quad = 1
    ReLU = 2

class SiLUType(enum.Enum):
    Default = 0
    Quad = 1
    ReLU = 2

class SoftmaxType(enum.Enum):
    Default = 0
    TwoQuad = 1
    LearnableTwoQuad = 2
    TwoReLU = 3

GeLUTypeMap = {
    "0": GeLUType.Default,
    "1": GeLUType.Quad,
    "2": GeLUType.ReLU,
}

SiLUTypeMap = {
    "0": SiLUType.Default,
    "1": SiLUType.Quad,
    "2": SiLUType.ReLU,
}

SoftmaxTypeMap = {
    "0": SoftmaxType.Default,
    "1": SoftmaxType.TwoQuad,
    "2": SoftmaxType.LearnableTwoQuad,
    "3": SoftmaxType.TwoReLU,
}

@contextmanager
def hack_embed_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    from jax_utils import HackEmbed
    # hijack some target functions
    raw_Embed = nn.Embed
    nn.Embed = HackEmbed
    yield
    # recover back
    nn.Embed = raw_Embed

def hack_softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    if softmax_type == SoftmaxType.Default:
        x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
        x = x - x_max

        # exp on large negative is clipped to zero
        b = x > -14
        nexp = jnp.exp(x) * b

        divisor = jnp.sum(nexp, axis, where=where, keepdims=True)
        print("using hack softmax default")
        return nexp / divisor
    # the cost of (x_i + c)^2 / sum_i (x_i + c)^2 is the same as the cost of x_i^2 / sum_i x_i^2
    elif softmax_type == SoftmaxType.TwoQuad:
        x_sq = jnp.square(x)
        divisor = jnp.sum(x_sq, axis, where=where, keepdims=True)
        print("using hack softmax 2Quad")
        return x_sq / divisor
    # the cost of (a_i * x_i + b_i)^2 / sum_i (a_i * x_i + b_i)^2 is the same as the cost of x_i^3 / sum_i x_i^3
    elif softmax_type == SoftmaxType.LearnableTwoQuad:
        x_ = jnp.multiply(x, x)
        x_sq_ = jnp.square(x_)
        divisor = jnp.sum(x_sq_, axis, where=where, keepdims=True)
        print("using hack softmax Learnable2Quad")
        return x_sq_ / divisor
    elif softmax_type == SoftmaxType.TwoReLU:
        x_relu = jnn.relu(x)
        divisor = jnp.sum(x_relu, axis, where=where, keepdims=True)
        print("using hack softmax TwoReLU")
        return x_relu / divisor
    else:
        raise NotImplementedError

@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax

def hack_gelu(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    if gelu_type == GeLUType.Default:
        b0 = x < -4.0
        b1 = x < -1.95
        b2 = x > 3.0
        b3 = b1 ^ b2 ^ True # x in [-1.95, 3.0]
        b4 = b0 ^ b1 # x in [-4, -1.95] 

        # seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
        # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
        a_coeffs = jnp.array([-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728])
        b_coeffs = jnp.array([0.008526321541038084,  0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187])
        x2 = jnp.square(x)
        x3 = jnp.multiply(x, x2)
        x4 = jnp.square(x2)
        x6 = jnp.square(x3)

        seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
        seg2 = b_coeffs[6] * x6 + b_coeffs[4] * x4 + b_coeffs[2] * x2 + b_coeffs[1] * x + b_coeffs[0]

        ret = b2 * x + b4 * seg1 + b3 * seg2
        print("using hack gelu default")

        return ret
    # GeLU(x) ≈ 0.125x^2 + 0.25x + 0.5
    elif gelu_type == GeLUType.Quad:
        coeffs = jnp.array([0.125, 0.25, 0.5])
        x2 = jnp.square(x)
        ret = coeffs[2] * x2 + coeffs[1] * x + coeffs[0]
        print("using hack gelu quad")
        return ret
    elif gelu_type == GeLUType.ReLU:
        ret = jnn.relu(x)
        print("using hack gelu relu")
        return ret
    else:
        raise NotImplementedError

@contextmanager
def hack_gelu_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_gelu = nn.gelu
    # nn.gelu = hack_gelu
    transformers.modeling_flax_utils.ACT2FN["gelu"] = hack_gelu
    transformers.modeling_flax_utils.ACT2FN["gelu_new"] = hack_gelu
    yield
    # recover back
    # nn.gelu = raw_gelu
    transformers.modeling_flax_utils.ACT2FN["gelu"] = raw_gelu
    transformers.modeling_flax_utils.ACT2FN["gelu_new"] = raw_gelu

def hack_silu(x: Array) -> Array:
    if silu_type == SiLUType.Default:
        b0 = x < -8.0
        b1 = x < -4.0
        b2 = x > 4.0
        b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
        b4 = b0 ^ b1  # x in [-8.0, -4.0)
        # seg1 =  a[2] * x^2 + a[1] * x + a[0]
        # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[0]
        a_coeffs = jnp.array(
            [-0.3067541139982155, -0.0819767021525476, -0.0055465625580307]
        )
        b_coeffs = jnp.array(
            [
                0.0085064025895951,
                0.5,
                0.2281430841728270,
                -0.011113046708173,
                0.0002743776353465,
            ]
        )
        x2 = jnp.square(x)
        x4 = jnp.square(x2)
        x6 = x2 * x4
        seg1 = a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
        seg2 = (
            b_coeffs[4] * x6
            + b_coeffs[3] * x4
            + b_coeffs[2] * x2
            + b_coeffs[1] * x
            + b_coeffs[0]
        )
        ret = b2 * x + b4 * seg1 + b3 * seg2
        print("using hack silu default")

        return ret
    elif silu_type == SiLUType.Quad:
        coeffs = jnp.array([0.125, 0.25, 0.5])
        x2 = jnp.square(x)
        ret = coeffs[2] * x2 + coeffs[1] * x + coeffs[0]
        print("using hack silu quad")
        return ret
    elif silu_type == SiLUType.ReLU:
        ret = jnn.relu(x)
        print("using hack silu relu")
        return ret
    else:
        raise NotImplementedError

@contextmanager
def hack_silu_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_silu = nn.silu
    nn.silu = hack_silu
    yield
    # recover back
    nn.silu = raw_silu

#############################################################################
# Benchmarking Logic
#############################################################################

def generation_through_API(input_ids, params, max_new_tokens):
    model = module(config=config)
    outputs = model.generate(input_ids=input_ids, params=params, max_new_tokens=max_new_tokens, do_sample=False, trace=True)
    return outputs

def prefilling(input_ids, params, max_new_tokens):
    model = module(config=config)
    gen_state, max_length = model.generate(input_ids=input_ids, params=params, max_new_tokens=max_new_tokens, do_sample=False, trace=True)
    return gen_state, max_length

# the GreedyState is input as components to allow private-public split
def decoding(cur_len, sequences, running_token, is_sent_finished, past_key_values, cache_index, attention_mask, position_ids, params, max_length):
    model = module(config=config)
    # this is the only component that needs to be public in past_key_values
    for key in past_key_values['transformer']['h'].keys():
        past_key_values['transformer']['h'][key][attn_key]['cache_index'] = cache_index
    model_kwargs = {
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    gen_state = GreedyState(
        cur_len=cur_len,
        sequences=sequences,
        running_token=running_token,
        is_sent_finished=is_sent_finished,
        model_kwargs=model_kwargs,
    )
    outputs = model._greedy_search(input_ids=None, max_length=max_length, params=params, state=gen_state)
    return outputs

def run_on_cpu(max_new_tokens=5):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(
        prompt, return_tensors='jax'
    )
    print(f"prompt_len: {input_ids.shape}")
    start = time.time()
    gen_state, max_length = prefilling(input_ids, init_params, max_new_tokens)
    end = time.time()
    print(f"Elapsed Time Prefilling (CPU): {end - start} seconds")
    start = time.time()
    past_key_values = gen_state.model_kwargs["past_key_values"]
    key_value_dict = jax.tree_util.tree_map(lambda x: type(x), past_key_values)
    print(f"key_value_dict: {key_value_dict}")
    cache_index = past_key_values['transformer']['h']['0'][attn_key]['cache_index']
    outputs_ids = decoding(cur_len=gen_state.cur_len, sequences=gen_state.sequences, running_token=gen_state.running_token, is_sent_finished=gen_state.is_sent_finished, past_key_values=past_key_values, cache_index=cache_index, attention_mask=gen_state.model_kwargs["attention_mask"], position_ids=gen_state.model_kwargs["position_ids"], params=init_params, max_length=max_length)
    end = time.time()
    print(f"Elapsed Time Decoding (CPU): {end - start} seconds")
    output_string = tokenizer.batch_decode(outputs_ids['sequences'], skip_special_tokens=True)
    return output_string

# def run_on_spu(max_new_tokens=5):
def spu_prefilling(max_new_tokens=5):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(
        prompt, return_tensors='jax'
    )
    print(f"prompt_len: {input_ids.shape}")

    import functools
    prefilling_partial = functools.partial(prefilling, max_new_tokens=max_new_tokens)
    if not simulation:
        start = time.time()
        input_ids = ppd.device("P1")(lambda x: x)(input_ids)
        params = ppd.device("P2")(lambda x: x)(init_params)
        end = time.time()
        print(f"Elapsed Time Prefilling (SPU Input): {end - start} seconds")
    with hack_softmax_context("hijack jax softmax", enabled = True), hack_gelu_context("hijack jax gelu", enabled=True), hack_silu_context("hijack jax silu", enabled=True), hack_embed_context("hijack jax embed", enabled=skip_embed):
        if not simulation:
            start = time.time()
            # output = ppd.device("SPU")(prefilling_partial, copts=copts)(input_ids, params)
            # gen_state, cur_len, max_length = ppd.get(output)
            outputs = ppd.device("SPU")(prefilling_partial, copts=copts)(input_ids, params)
            end = time.time()
            print(f"Elapsed Time Prefilling (SPU Protocol): {end - start} seconds")
            gen_state, max_length = ppd.get(outputs)
        else:
            start = time.time()
            spu_prefilling = pps.sim_jax(simulator, prefilling_partial, copts=copts)
            gen_state, max_length = spu_prefilling(input_ids, init_params)
            end = time.time()
            print(f"Elapsed Time Prefilling (SPU Simulation): {end - start} seconds")
    return
    '''
    past_key_values = gen_state.model_kwargs["past_key_values"]
    cache_index = past_key_values['transformer']['h']['0'][attn_key]['cache_index']
    decoding_partial = functools.partial(decoding, cur_len=gen_state.cur_len, is_sent_finished=gen_state.is_sent_finished, attention_mask=gen_state.model_kwargs["attention_mask"], cache_index=cache_index, position_ids=gen_state.model_kwargs["position_ids"], max_length=max_length)
    if not simulation:
        start = time.time()
        sequences = ppd.device("P1")(lambda x: x)(gen_state.sequences)
        running_token = ppd.device("P1")(lambda x: x)(gen_state.running_token)
        past_key_values = ppd.device("P1")(lambda x: x)(past_key_values)
        end = time.time()
        print(f"Elapsed Time Decoding (SPU Input): {end - start} seconds")
    with hack_softmax_context("hijack jax softmax", enabled = True), hack_gelu_context("hijack jax gelu", enabled=True), hack_silu_context("hijack jax silu", enabled=True), hack_embed_context("hijack jax embed", enabled=True):
        if not simulation:
            start = time.time()
            outputs_ids = ppd.device("SPU")(decoding_partial, copts=copts)(sequences=sequences, running_token=running_token, past_key_values=past_key_values, params=params)
            outputs_ids = ppd.get(outputs_ids)
            end = time.time()
            print(f"Elapsed Time Decoding (SPU Protocol): {end - start} seconds")
        else:
            start = time.time()
            spu_decoding = pps.sim_jax(simulator, decoding_partial, copts=copts)
            outputs_ids = spu_decoding(sequences=gen_state.sequences, running_token=gen_state.running_token, past_key_values=past_key_values, params=init_params)
            end = time.time()
            print(f"Elapsed Time Decoding (SPU Simulation): {end - start} seconds")

    output_string = tokenizer.batch_decode(outputs_ids['sequences'], skip_special_tokens=True)
    return output_string
    '''

def spu_decoding(max_new_tokens=5):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(
        prompt, return_tensors='jax'
    )
    print(f"prompt_len: {input_ids.shape}")

    import functools
    gen_state, max_length = prefilling(input_ids, init_params, max_new_tokens)
    past_key_values = gen_state.model_kwargs["past_key_values"]
    cache_index = past_key_values['transformer']['h']['0'][attn_key]['cache_index']
    decoding_partial = functools.partial(decoding, cur_len=gen_state.cur_len, is_sent_finished=gen_state.is_sent_finished, attention_mask=gen_state.model_kwargs["attention_mask"], cache_index=cache_index, position_ids=gen_state.model_kwargs["position_ids"], max_length=max_length)
    if not simulation:
        start = time.time()
        sequences = ppd.device("P1")(lambda x: x)(gen_state.sequences)
        running_token = ppd.device("P1")(lambda x: x)(gen_state.running_token)
        past_key_values = ppd.device("P1")(lambda x: x)(past_key_values)
        params = ppd.device("P2")(lambda x: x)(init_params)
        end = time.time()
        print(f"Elapsed Time Decoding (SPU Input): {end - start} seconds")
    with hack_softmax_context("hijack jax softmax", enabled = True), hack_gelu_context("hijack jax gelu", enabled=True), hack_silu_context("hijack jax silu", enabled=True), hack_embed_context("hijack jax embed", enabled=skip_embed):
        if not simulation:
            start = time.time()
            outputs_ids = ppd.device("SPU")(decoding_partial, copts=copts)(sequences=sequences, running_token=running_token, past_key_values=past_key_values, params=params)
            end = time.time()
            outputs_ids = ppd.get(outputs_ids)
            print(f"Elapsed Time Decoding (SPU Protocol): {end - start} seconds")
        else:
            start = time.time()
            spu_decoding = pps.sim_jax(simulator, decoding_partial, copts=copts)
            outputs_ids = spu_decoding(sequences=gen_state.sequences, running_token=gen_state.running_token, past_key_values=past_key_values, params=init_params)
            end = time.time()
            print(f"Elapsed Time Decoding (SPU Simulation): {end - start} seconds")

    output_string = tokenizer.batch_decode(outputs_ids['sequences'], skip_special_tokens=True)
    return output_string

def spu_generate(max_new_tokens=5):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(
        prompt, return_tensors='jax'
    )
    print(f"prompt_len: {input_ids.shape}")

    import functools
    text_gen = functools.partial(generation_through_API, max_new_tokens=max_new_tokens)
    if not simulation:
        start = time.time()
        input_ids = ppd.device("P1")(lambda x: x)(input_ids)
        params = ppd.device("P2")(lambda x: x)(init_params)
        end = time.time()
        print(f"Elapsed Time Generate (SPU Input): {end - start} seconds")
    with hack_softmax_context("hijack jax softmax", enabled = True), hack_gelu_context("hijack jax gelu", enabled=True), hack_silu_context("hijack jax silu", enabled=True), hack_embed_context("hijack jax embed", enabled=skip_embed):
        if not simulation:
            start = time.time()
            outputs = ppd.device("SPU")(text_gen, copts=copts)(input_ids, params)
            end = time.time()
            outputs = ppd.get(outputs)
            print(f"Elapsed Time Generate (SPU Protocol): {end - start} seconds")
        else:
            start = time.time()
            spu_text_gen = pps.sim_jax(simulator, text_gen, copts=copts)
            outputs = spu_text_gen(input_ids, init_params)
            end = time.time()
            print(f"Elapsed Time Prefilling (SPU Simulation): {end - start} seconds")

    # output_string = tokenizer.batch_decode(outputs_ids['sequences'], skip_special_tokens=True)
    # return output_string

if __name__ == '__main__':
    global module, init_params, prompt, tokenizer, config, simulation, simulator, attn_key, skip_embed, skip_block, skip_lm
    global gelu_type, silu_type, softmax_type

    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-m", "--model-name", default="llama3b")
    parser.add_argument("-c", "--config", default="./3pc.json")
    parser.add_argument("-s", "--seq-len", default=8)
    # any more than 1 token fails on llama7b
    parser.add_argument("-n", "--new-tokens", default=2)
    # parser.add_argument("-f", "--frozen", default=0)
    parser.add_argument("-H", "--head-compress", default=1)
    parser.add_argument("-p", "--head-pruning", default=0.0)
    parser.add_argument("-l", "--use-lora", default="False")
    parser.add_argument("-r", "--lora-rank", default=4)
    parser.add_argument("-E", "--skip-embed", default="True")
    # parser.add_argument("-B", "--skip-block", default="False")
    parser.add_argument("-L", "--skip-lm", default="True")
    parser.add_argument("-S", "--simulation", default="False")
    parser.add_argument("-ga", "--gelu-approx", default="0")
    parser.add_argument("-sa", "--softmax-approx", default="0")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    model_name = args.model_name
    seq_len = int(args.seq_len)
    new_tokens = int(args.new_tokens)
    # frozen = int(args.frozen)
    head_compress = int(args.head_compress)
    use_lora = args.use_lora == "True"
    lora_rank = int(args.lora_rank)
    simulation = args.simulation == "True"
    skip_embed = args.skip_embed == "True"
    skip_block = False #args.skip_block == "True"
    skip_lm = args.skip_lm == "True"
    # proxy for both activations
    gelu_type = GeLUTypeMap[args.gelu_approx]
    silu_type = SiLUTypeMap[args.gelu_approx]
    softmax_type = SoftmaxTypeMap[args.softmax_approx]
    head_pruning = float(args.head_pruning)

    if not simulation:
        ppd.init(conf["nodes"], conf["devices"])
    else:
        # config = conf["devices"]["SPU"]["config"]["runtime_config"]
        protocol = spu.ProtocolKind.REF2K
        num_parties = 1
        field = spu.FieldType.FM64
        sim_config = spu.RuntimeConfig(protocol=protocol, field=field)
        sim_config.enable_pphlo_profile = True
        sim_config.enable_hal_profile = True
        sim_config.fxp_exp_mode = 0
        sim_config.fxp_exp_iters = 5
        simulator = pps.Simulator(num_parties, sim_config)

    print(f"model name: {model_name}, seq len: {seq_len}, new tokens: {new_tokens}, head compress: {head_compress}, use lora: {use_lora}, lora rank: {lora_rank}, head_pruning: {head_pruning}, skip_embed: {skip_embed}, skip_block: {skip_block}, skip_lm: {skip_lm}")
    print(f"gelu_type: {gelu_type}, silu_type: {silu_type}, softmax_type: {softmax_type}")
    print(f"running on simulation? {simulation}")

    if model_name == "gpt2":
        import modeling_flax_gpt2
        modeling_flax_gpt2.USE_LORA = use_lora
        modeling_flax_gpt2.RANK = lora_rank

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = transformers.AutoConfig.from_pretrained(model_name)
        # config.n_layer = config.n_layer - frozen
        if head_pruning > 0.0:
            assert(head_compress == 1)
            config.head_pruning = True
            config.n_head = int(config.n_head * (1 - head_pruning))
            config.pruned_hidden_size = int(config.hidden_size * (1 - head_pruning))
        else:
            config.head_pruning = None
        config.n_layer = 1
        assert(config.n_head % head_compress == 0)
        config.n_head = config.n_head // head_compress
        config.tie_word_embeddings = False

        module = modeling_flax_gpt2.FlaxGPT2LMHeadModel

        attn_key = "attn"
    elif "llama" in model_name:
        import modeling_flax_llama7b
        modeling_flax_llama7b.USE_LORA = use_lora
        modeling_flax_llama7b.RANK = lora_rank

        tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b_v2")

        # use default config corresponding to the 7B model
        config = modeling_flax_llama7b.LLaMAConfig()
        if model_name == "llama3b":
            config.update(modeling_flax_llama7b.LLAMA_STANDARD_CONFIGS['3b'])
        # config.num_hidden_layers = config.num_hidden_layers - frozen
        if head_pruning > 0.0:
            assert(head_compress == 1)
            config.head_pruning = True
            config.num_attention_heads = int(config.num_attention_heads * (1 - head_pruning))
            config.pruned_hidden_size = int(config.hidden_size * (1 - head_pruning))
        else:
            config.head_pruning = None
        config.num_hidden_layers = 1
        assert(config.num_attention_heads % head_compress == 0)
        config.num_attention_heads = config.num_attention_heads // head_compress
        config.tie_word_embeddings = False

        module = modeling_flax_llama7b.FlaxLLaMAForCausalLM

        attn_key = "attention"
    config.skip_lm = skip_lm

    print(config)

    # prompt = 'Machine learning is great for humanity. It helps'
    prompt = 'the ' * (seq_len - 1)

    model = module(config=config)
    key1, key2 = jax.random.split(jax.random.key(0))
    input_shape = (1, 1)
    dummy_input = jax.random.normal(key1, input_shape)
    init_params = model.init_weights(rng=jax.random.PRNGKey(0), input_shape=input_shape)
    init_params_dict = jax.tree_util.tree_map(lambda x: x.shape, init_params)
    # param_dict = jax.tree_util.tree_map(lambda x: x.shape, pretrained_model.params['transformer']['h']['0'])
    # print(f"old_params: {json.dumps(param_dict, sort_keys=True, indent=2)}")
    print(f"new_params: {json.dumps(init_params_dict, sort_keys=True)}")
    # print(f"pretrained_model.params.keys(): {pretrained_model.params['transformer']['h']['0']['attn'].keys()}, parameter: {pretrained_model.params['transformer']['h']['0']['attn']['c_attn']}")
    # print('\n------\nRun on CPU')
    # output_string = run_on_cpu(new_tokens)
    # print(output_string)
    print('\n------\nRun on SPU')
    spu_prefilling(new_tokens)
    spu_decoding(new_tokens)
    # spu_generate(new_tokens)
    # output_string = run(use_spu=True)
    # output_string = run_on_spu(new_tokens)
    # print(output_string)