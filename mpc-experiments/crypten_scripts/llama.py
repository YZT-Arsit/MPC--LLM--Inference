import time
import math

import torch
import torch.nn.functional as F

import crypten
import crypten.nn as cnn
import crypten.communicator as comm
from crypten.common.functions import maximum

from utils import softmax_2RELU, softmax_2QUAD, softmax_L2QUAD, activation_quad, activation_newGeLU, activation_silu, encrypt_tensor

def memory_allocated():
    return_string = f"({torch.cuda.memory_allocated() / (1 << 30)} GiB, {torch.cuda.max_memory_allocated() / (1 << 30)} GiB)"
    torch.cuda.reset_max_memory_allocated()
    return return_string

def chunked_matmul(A, B, chunk_size):
    if A.shape[-1] != B.shape[-2]:
        raise ValueError("The inner dimensions must match for matrix multiplication.")

    if A.shape[-2] > B.shape[-1]:
        chunk_A = True
        dim = A.shape[-2]
        cat = cnn.Concat(dimension=-2)
    else:
        chunk_A = False
        dim = B.shape[-1]
        cat = cnn.Concat(dimension=-1)
    # Iterate through the chunks
    for i in range(0, dim, chunk_size):
        # Determine the size of the current chunk
        end = min(i + chunk_size, dim)
        
        # Perform the chunked matrix multiplication
        if chunk_A:
            tmp = A[..., i:end, :].matmul(B)
        else:
            tmp = A.matmul(B[..., i:end])
        if i == 0:
            result = tmp
        else:
            result = cat([result, tmp])

    return result

class HackLinear(cnn.Module):
    def __init__(self, in_features, out_features, lora, lora_dim, pruneFactor = 1):
        super(HackLinear, self).__init__() 
        self.lora = lora
        self.lora_dim = lora_dim
        self.in_features = in_features
        self.out_features = out_features
        self.pruneFactor = pruneFactor
        if self.lora:
            self.A = cnn.Linear(in_features, self.lora_dim)
            self.B = cnn.Linear(self.lora_dim, out_features)
        else:
            self.tokenSubDim = out_features // self.pruneFactor
            self.lastTokenDim = out_features - (self.pruneFactor - 1) * self.tokenSubDim
            self.moduleList = []

            for _ in range(self.pruneFactor - 1):
                ll = cnn.Linear(in_features, self.tokenSubDim)
                self.moduleList.append(ll)

            self.moduleList.append(cnn.Linear(in_features, self.lastTokenDim))
            self.cat = cnn.Concat(dimension=-1)

    def cuda(self, device=None):
        super(HackLinear, self).cuda(device=device)

        if self.lora:
            self.A.cuda(device=device)
            self.B.cuda(device=device)
        else:
            for i in range(len(self.moduleList)):
                self.moduleList[i].cuda(device=device)
        return self

    def encrypt(self, mode=True, src=0):
        super(HackLinear, self).encrypt(mode=mode, src=src)

        if self.lora:
            self.A.encrypt(mode=mode, src=src)
            self.B.encrypt(mode=mode, src=src)
        else:
            for i in range(len(self.moduleList)):
                self.moduleList[i].encrypt(mode=mode, src=src)
        return self

    def forward(self, hidden_states):
        res = None
        
        if self.lora:
            res = self.B(self.A(hidden_states))
        else:
            for i, ll in enumerate(self.moduleList):
                if i == 0:
                    res = ll(hidden_states)
                else:
                    res = self.cat([res, ll(hidden_states)])
        print(f"Linear shape: {self.in_features} x {self.out_features}, LoRA: {self.lora_dim if self.lora else 'NA'}, Prune Factor: {self.pruneFactor}, GPU memory: {memory_allocated()}")
            
        return res


class llama(cnn.Module):
    def __init__(self, config, timing):
        super(llama, self).__init__()
        self.config = config

        # No need to init weight for timing purpose
        self.embeddings = gptEmbeddings(config, timing)
        self.encoder = cnn.ModuleList([gptLayer(config, timing) for _ in range(config.num_hidden_layers)])
        self.lm_head = lm_head(config, timing) #cnn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.smax = cnn.Softmax(dim=-1)
        self.cat = cnn.Concat(dimension=1)
        self.layer_norm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.timing = timing
   
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
 
    def forward(self, input_ids, past_list):
        output = self.embeddings(input_ids)
        t0 = time.time()
        for layer_id, layer in enumerate(self.encoder):
            # pass in a past key/value of shape [[b, s, h], [b, s, h]] !!not tuple, it will get deep copied..!!
            if len(past_list[layer_id]) == 0:
                print(f"input to layer {layer_id} None")
            else:
                print("input to layer size: ", past_list[layer_id][0].shape, past_list[layer_id][1].shape)
            #output, past = layer(output, past_list[layer_id])
            output = layer(output, past_list[layer_id])
            #past_list[layer_id].append()
        t1 = time.time()
        self.timing["LayerTime"] += (t1-t0)
        output = self.layer_norm(output)
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        output = self.lm_head(output)
        comm1 = comm.get().get_communication_stats()
        t1 = time.time()
        self.timing["LinearTime"] += (t1-t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        self.timing["lmHeadTime"] += (t1-t0)
        self.timing["lmHeadCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["lmHeadCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return output#, past

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,s,v)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generation_time = {}
        past_list = [[] for _ in range(self.config.num_hidden_layers)]
        generation_stage = False
        for token_id in range(max_new_tokens):
            t_start = time.time()
            b, s, _ = idx.shape
            time_s = time.time()
            comm0 = comm.get().get_communication_stats()
            # if the sequence context is growing too long we must crop it at max_position_embeddings
            idx_cond = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:,:]
            if not generation_stage:
                logits = self(idx_cond, past_list)
                generation_stage = True
            else:
                logits = self(idx_cond[:, -1:, :], past_list)
            t1 = time.time()
            comm1 = comm.get().get_communication_stats()
            self.timing["GenerateMainTime"] += (t1-t_start)
            self.timing["GenerateMainCommTime"] += (comm1["time"] - comm0["time"])
            self.timing["GenerateMainCommByte"] += (comm1["bytes"] - comm0["bytes"])
            t0 = time.time()
            comm0 = comm.get().get_communication_stats()
            logits = logits[:, -1:, :] / temperature
            probs = self.smax(logits)
            idx_next = maximum.argmax(probs, dim=-1)
            idx = self.cat([idx, idx_next])
            comm1 = comm.get().get_communication_stats()
            t1 = time.time()
            time_e = time.time()
            generation_time.update({(b, s): time_e - time_s})
            self.timing["GenerateOtherTime"] += (t1-t0)
            self.timing["GenerateOtherCommTime"] += (comm1["time"] - comm0["time"])
            self.timing["GenerateOtherCommByte"] += (comm1["bytes"] - comm0["bytes"])
            self.timing["cur_step_total"] = (time.time() - t_start)
            print(f"{token_id}: {self.timing}")
            self.reset_timing()
            print(generation_time)
        return idx

class LlamaRMSNorm(cnn.Module):
    def __init__(self, hidden_size, eps):
        super(LlamaRMSNorm, self).__init__()
        self.register_parameter("weight", torch.ones(hidden_size))
        self.variance_epsilon = torch.tensor([eps]).item()
        self.one = torch.tensor([1.0]).item()
        self.two = torch.tensor([2.0]).item()
        self.neg_one = torch.tensor([-1.0]).item()
        self.pow = cnn.Pow()
        self.mean = cnn.Mean(dim=2, keepdim=True)
        self.sqrt = cnn.Sqrt()
    
    def forward(self, hidden_states):
        powed = self.pow((hidden_states, self.two))
        variance = self.mean(powed)
        sqrt = self.pow((self.sqrt(variance + self.variance_epsilon), self.neg_one))
        return hidden_states * sqrt * self.weight

class LlamaRotaryEmbedding(cnn.Module):
    # For easy implmentation, we disable the logic in rotary, to use a constant cos/sin.
    # This has the same runtime.
    def __init__(self, bs, max_seq_len, num_attention_heads, head_size):
        super().__init__()
        self.register_parameter("weight", torch.ones(bs, num_attention_heads, max_seq_len, head_size))
        self.cat = cnn.Concat(dimension=-1)
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return self.cat([x1, x2])

    def forward(self, q, k):
        cos = sin = self.weight[:,:, :q.shape[2], :]
        # print(q.shape, cos.shape)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
        

class lm_head(cnn.Module):
    def __init__(self, config, timing):
        super(lm_head, self).__init__() 
        # save memory
        pruneFactor = 128
        self.linear = HackLinear(config.hidden_size, config.vocab_size, False, -1, pruneFactor)
        '''
        self.tokenSubDim = config.vocab_size // self.pruneFactor
        self.lastTokenDim = config.vocab_size - (self.pruneFactor - 1) * self.tokenSubDim
        self.moduleList = []

        for _ in range(self.pruneFactor - 1):
            ll = cnn.Linear(config.hidden_size, self.tokenSubDim)
            self.moduleList.append(ll)

        self.moduleList.append(cnn.Linear(config.hidden_size, self.lastTokenDim))
        self.cat = cnn.Concat(dimension=-1)
        '''

        self.config = config
        self.timing = timing

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def cuda(self, device=None):
        super(lm_head, self).cuda(device=device)

        self.linear.cuda(device=device)
        '''
        for i in range(len(self.moduleList)):
            self.moduleList[i].cuda(device=device)
        '''
        return self

    def encrypt(self, mode=True, src=0):
        super(lm_head, self).encrypt(mode=mode, src=src)

        self.linear.encrypt(mode=mode, src=src)
        '''
        for i in range(len(self.moduleList)):
            self.moduleList[i].encrypt(mode=mode, src=src)
        '''
        return self

    def forward(self, hidden_states):
        res = self.linear(hidden_states)
        '''
        for i, ll in enumerate(self.moduleList):
            if i == 0:
                res = ll(hidden_states)
            else:
                res = self.cat([res, ll(hidden_states)])
        '''
            
        return res

class gptEmbeddings(cnn.Module):
    def __init__(self, config, timing):
        super(gptEmbeddings, self).__init__()
        # save memory
        # pruneFactor = 128
        # self.linear = HackLinear(config.vocab_size, config.hidden_size, False, -1, pruneFactor)
        #'''
        self.pruneFactor = 128
        self.tokenSubDim = config.vocab_size // self.pruneFactor
        self.lastTokenDim = config.vocab_size - (self.pruneFactor - 1) * self.tokenSubDim
        self.moduleList = []

        for _ in range(self.pruneFactor - 1):
            ll = cnn.Linear(self.tokenSubDim, config.hidden_size)
            self.moduleList.append(ll)

        self.moduleList.append(cnn.Linear(self.lastTokenDim, config.hidden_size))
        #'''

        # self.wpe = cnn.Linear(config.max_position_embeddings, config.hidden_size)
        # print(config.hidden_size)
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.timing = timing

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def cuda(self, device=None):
        super(gptEmbeddings, self).cuda(device=device)

        # self.linear.cuda(device=device)
        #'''
        for i in range(len(self.moduleList)):
            self.moduleList[i].cuda(device=device)
        #'''
        # self.wpe.cuda(device=device)
        return self

    def encrypt(self, mode=True, src=0):
        super(gptEmbeddings, self).encrypt(mode=mode, src=src)

        # self.linear.encrypt(mode=mode, src=src)
        #'''
        for i in range(len(self.moduleList)):
            self.moduleList[i].encrypt(mode=mode, src=src)
        #'''
        # self.wpe.encrypt(mode=mode, src=src)
        return self

    def forward(self, input_ids):
        embeddings = None
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        # embeddings = self.linear(input_ids)
        #'''
        for i, ll in enumerate(self.moduleList):
            #print(ll.weight.shape)
            if i != (len(self.moduleList) - 1):
            #   print(input_ids[:, :, i * self.tokenSubDim : (i + 1) * self.tokenSubDim].shape)
                res = ll(input_ids[:, :, i * self.tokenSubDim : (i + 1) * self.tokenSubDim])
            else:
                res = ll(
                    input_ids[
                        :,:,
                        i * self.tokenSubDim : i * self.tokenSubDim + self.lastTokenDim
                    ]
                )

            embeddings = res if embeddings is None else (embeddings + res)
        #'''
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["EmbedTime"] += (t1-t0)
        self.timing["EmbedCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["EmbedCommByte"] += (comm1["bytes"] - comm0["bytes"])
        #print("benchmarking embed: ", input_ids.shape, t1-t0)

       # position_embeddings = (self.wpe.weight[:,:input_ids.shape[1]]).transpose(0,1)
     #   print(position_embeddings.shape, self.position_embeddings.weight.shape)
       # position_embeddings = position_embeddings.repeat(input_ids.shape[0],1,1)
     #   print(position_embeddings.shape, embeddings.shape)
       # embeddings += position_embeddings

        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        orig_size = embeddings.size()
        # embeddings = embeddings.view(-1, self.config.hidden_size)
        embeddings = self.LayerNorm(embeddings).view(orig_size)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["NormTime"] += (t1-t0)
        self.timing["NormCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["NormCommByte"] += (comm1["bytes"] - comm0["bytes"])
        embeddings = self.dropout(embeddings)
        return embeddings

class gptLayer(cnn.Module):
    def __init__(self, config, timing):
        super(gptLayer, self).__init__()
        self.config = config
        self.attention = gptAttention(config, timing)
        self.intermediate = gptIntermediate(config, timing)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.embedding = LlamaRotaryEmbedding(config.batch_size, config.max_position_embeddings, config.num_attention_heads, config.hidden_size // config.num_attention_heads)
        self.output = gptOutput(config, timing)
        self.config = config
        self.timing = timing
 
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
 
    def forward(self, hidden_states, past):
        #attention_output, past = self.attention(hidden_states, past)
        #print("debug copy before: ", past)
        residual = hidden_states
        # print(hidden_states.shape)
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(hidden_states, past)
        attention_output = residual + attention_output
        
        residual = hidden_states
        attention_output = self.post_attention_layernorm(attention_output)
        #print("debug copy after: ", past)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        layer_output = residual + layer_output
        return layer_output#, past
        
class gptAttention(cnn.Module):
    def __init__(self, config, timing):
        super(gptAttention, self).__init__()
        self.self = gptSelfAttention(config, timing)
        self.output = gptSelfOutput(config, timing)
        
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def forward(self, hidden_states, past):
        #self_output, past = self.self(hidden_states, past)
        self_output = self.self(hidden_states, past)
        attention_output = self.output(self_output, hidden_states)
        return attention_output#, past

class gptSelfAttention(cnn.Module):
    def __init__(self, config, timing):
        super(gptSelfAttention, self).__init__()
        
        self.num_attention_heads = config.num_attention_heads
        if config.head_pruning is not None:
            self.hidden_size = config.pruned_hidden_size
        else:
            self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.embedding = LlamaRotaryEmbedding(config.batch_size, config.max_position_embeddings, config.num_attention_heads, self.hidden_size // config.num_attention_heads)
        
        self.config = config
        pruneFactor = 4
        self.query = HackLinear(config.hidden_size, self.hidden_size, config.lora, config.lora_dim, pruneFactor)
        self.key = HackLinear(config.hidden_size, self.hidden_size, config.lora, config.lora_dim, pruneFactor)
        self.value = HackLinear(config.hidden_size, self.hidden_size, config.lora, config.lora_dim, pruneFactor)
        '''
        if config.lora:
            self.query_A = cnn.Linear(self.hidden_size, config.lora_dim)
            self.query_B = cnn.Linear(config.lora_dim, self.hidden_size)
            self.key_A = cnn.Linear(self.hidden_size, config.lora_dim)
            self.key_B = cnn.Linear(config.lora_dim, self.hidden_size)
            self.value_A = cnn.Linear(self.hidden_size, config.lora_dim)
            self.value_B = cnn.Linear(config.lora_dim, self.hidden_size)
        else:
            self.query = cnn.Linear(self.hidden_size, self.hidden_size)
            self.key = cnn.Linear(self.hidden_size, self.hidden_size)
            self.value = cnn.Linear(self.hidden_size, self.hidden_size)
        '''

        self.cat = cnn.Concat(dimension=-2)
        # TODO: implement causal mask
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size))
        self.dropout = cnn.Dropout(config.attention_probs_dropout_prob)
        if config.softmax_act == "softmax":
            self.smax = cnn.Softmax(dim=-1)
        elif config.softmax_act == "softmax_2RELU":
            self.smax = softmax_2RELU(dim=-1)
        elif config.softmax_act == "softmax_2QUAD":
            self.smax = softmax_2QUAD(dim=-1)
        elif config.softmax_act == "softmax_L2QUAD":
            self.smax = softmax_L2QUAD(dim=-1)
        else:
            raise ValueError(f"softmax type {config.softmax_act} not implemented.")
        self.timing = timing
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, past):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        '''
        if self.config.lora:
            query_layer = self.transpose_for_scores(self.query_B(self.query_A(hidden_states)))
        else:
            query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        if self.config.lora:
            key_layer = self.transpose_for_scores(self.key_B(self.key_A(hidden_states)))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
        print("key shape:", key_layer.shape)
        if self.config.lora:
            value_layer = self.transpose_for_scores(self.value_B(self.value_A(hidden_states)))
        else:
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        '''
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        print("key shape:", key_layer.shape)
        
        if len(past) != 0:
            past_key, past_value = past
            print("cat debug: ", past_key.shape, key_layer.shape )
            key_layer = self.cat([past_key, key_layer])
            value_layer = self.cat([past_value, value_layer])
            past[0] = key_layer
            past[1] = value_layer
        else:
            # update past
            past.append(key_layer)
            past.append(value_layer)        
        
        query_layer, key_layer = self.embedding(query_layer, key_layer)
           
        decoding = query_layer.shape[-2] == 1
        if decoding:
            attention_scores = query_layer.matmul(key_layer.transpose(-1, -2))
            print(f"Matmul shape: {query_layer.shape} x {key_layer.shape}, GPU memory: {memory_allocated()}")
        else: # prefilling
            pruneFactor = 4
            chunk_size = key_layer.shape[-2] // pruneFactor
            attention_scores = chunked_matmul(query_layer, key_layer.transpose(-1, -2), chunk_size)
            print(f"Matmul shape: {pruneFactor} x {query_layer.shape} x {key_layer.shape}/{pruneFactor}, GPU memory: {memory_allocated()}")

        #print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # TODO: implement mask
        # attention_scores = attention_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        #print("smax operands: ", attention_scores.shape)
        if decoding:
            attention_probs = self.smax(attention_scores)
            print(f"Softmax shape: {attention_scores.shape}, GPU memory: {memory_allocated()}")
        else:
            pruneFactor = 8
            dim_2 = attention_scores.shape[-2]
            assert dim_2 % pruneFactor == 0
            chunk = dim_2 // pruneFactor
            for i in range(pruneFactor):
                if i == 0:
                    attention_probs = self.smax(attention_scores[..., i * chunk : (i + 1) * chunk, :])
                else:
                    tmp = self.smax(attention_scores[..., i * chunk : (i + 1) * chunk, :])
                    attention_probs = self.cat([attention_probs, tmp])
                    del tmp
            print(f"Softmax shape: {pruneFactor} x ({attention_probs.shape} / {pruneFactor}), GPU memory: {memory_allocated()}")
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["SoftmaxTime"] += (t1 - t0)
        self.timing["SoftmaxCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["SoftmaxCommByte"] += (comm1["bytes"] - comm0["bytes"])

        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        # context_layer = attention_probs.matmul(value_layer)
        if decoding:
            context_layer = attention_probs.matmul(value_layer)
            print(f"Matmul shape: {attention_probs.shape} x {value_layer.shape}, GPU memory: {memory_allocated()}")
        else: # prefilling
            pruneFactor = 4
            chunk_size = attention_probs.shape[-1] // pruneFactor
            context_layer = chunked_matmul(attention_probs, value_layer, chunk_size)
            print(f"Matmul shape: {pruneFactor} x {attention_probs.shape} x {value_layer.shape}/{pruneFactor}, GPU memory: {memory_allocated()}")
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])

        context_layer = context_layer.permute(0, 2, 1, 3)#.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        del query_layer, attention_scores, attention_probs
        #print("debug shapes after attention: ", context_layer.shape, key_layer.shape, value_layer.shape)        
        return context_layer#, (key_layer, value_layer)

class gptSelfOutput(cnn.Module):
    def __init__(self, config, timing):
        super(gptSelfOutput, self).__init__()
        # self.dense = cnn.Linear(config.hidden_size, config.hidden_size)
        pruneFactor = 4
        if config.head_pruning is not None:
            self.dense = HackLinear(config.pruned_hidden_size, config.hidden_size, config.lora, config.lora_dim, pruneFactor)
        else:
            self.dense = HackLinear(config.hidden_size, config.hidden_size, config.lora, config.lora_dim, pruneFactor)
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.timing = timing
        self.config = config
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def forward(self, hidden_states, input_tensor):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states = self.dense(hidden_states)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        
        hidden_states = self.dropout(hidden_states)
        # residual connection here
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        orig_size = hidden_states.size()
        hidden_states = hidden_states + input_tensor
        # hidden_states = hidden_states.view(-1, self.config.hidden_size)
        hidden_states = self.LayerNorm(hidden_states).view(orig_size)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["NormTime"] += (t1 - t0)
        self.timing["NormCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["NormCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return hidden_states

class gptIntermediate(cnn.Module):
    def __init__(self, config, timing):
        super(gptIntermediate, self).__init__()
        '''
        if config.lora:
            self.gate_proj_A = cnn.Linear(config.hidden_size, config.lora_dim)
            self.gate_proj_B = cnn.Linear(config.lora_dim, config.intermediate_size)
            self.up_proj_A = cnn.Linear(config.hidden_size, config.lora_dim)
            self.up_proj_B = cnn.Linear(config.lora_dim, config.intermediate_size)
        else:
            self.gate_proj = cnn.Linear(config.hidden_size, config.intermediate_size)
            self.up_proj = cnn.Linear(config.hidden_size, config.intermediate_size)
        '''
        pruneFactor = 16
        self.gate_proj = HackLinear(config.hidden_size, config.intermediate_size, config.lora, config.lora_dim, pruneFactor)
        self.up_proj = HackLinear(config.hidden_size, config.intermediate_size, config.lora, config.lora_dim, pruneFactor)
        if config.hidden_act == "relu":
            self.intermediate_act_fn = cnn.ReLU()
        elif config.hidden_act == "quad":
            self.intermediate_act_fn = activation_quad()
        elif config.hidden_act == "newGeLU":
            self.intermediate_act_fn = activation_newGeLU()
        elif config.hidden_act == "silu":
            self.intermediate_act_fn = activation_silu()
        else:
            raise ValueError(f"activation type {config.hidden_act} not implemented")
        self.timing = timing
        self.config = config

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def forward(self, hidden_states):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states_gate = self.gate_proj(hidden_states)
        hidden_states_up = self.up_proj(hidden_states)
        '''
        if self.config.lora:
            hidden_states_gate = self.gate_proj_B(self.gate_proj_A(hidden_states))
            hidden_states_up = self.up_proj_B(self.up_proj_A(hidden_states))
        else:
            hidden_states_gate = self.gate_proj(hidden_states)
            hidden_states_up = self.up_proj(hidden_states)
        '''
        hidden_states = hidden_states_gate * hidden_states_up
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states = self.intermediate_act_fn(hidden_states)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["ActTime"] += (t1 - t0)
        self.timing["ActCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["ActCommByte"] += (comm1["bytes"] - comm0["bytes"])
        del hidden_states_gate, hidden_states_up
        return hidden_states


class gptOutput(cnn.Module):
    def __init__(self, config, timing):
        super(gptOutput, self).__init__()
        pruneFactor = 16
        self.dense = HackLinear(config.intermediate_size, config.hidden_size, config.lora, config.lora_dim, pruneFactor)
        '''
        if config.lora:
            self.dense_A = cnn.Linear(config.intermediate_size, config.lora_dim)
            self.dense_B = cnn.Linear(config.lora_dim, config.hidden_size)
        else:
            self.dense = cnn.Linear(config.intermediate_size, config.hidden_size)
        '''
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.timing = timing
        self.config = config
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def forward(self, hidden_states, input_tensor):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states = self.dense(hidden_states)
        '''
        if self.config.lora:
            hidden_states = self.dense_B(self.dense_A(hidden_states))
        else:
            hidden_states = self.dense(hidden_states)
        '''
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        hidden_states = self.dropout(hidden_states)
        # residual connection
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        orig_size = hidden_states.size()
        hidden_states = hidden_states + input_tensor
        # hidden_states = hidden_states.view(-1, self.config.hidden_size)
        hidden_states = self.LayerNorm(hidden_states).view(orig_size)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["NormTime"] += (t1 - t0)
        self.timing["NormCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["NormCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return hidden_states

