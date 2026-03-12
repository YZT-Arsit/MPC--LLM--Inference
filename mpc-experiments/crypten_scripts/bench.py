import sys
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model
import argparse

# needed on MacOS to avoid serialization issues with spawn
import multiprocessing
multiprocessing.set_start_method('fork')

from llama import llama

# Inference arguments
class llama3b_config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 26
       self.hidden_size = 3200
       self.intermediate_size = 8640
       self.max_position_embeddings = 2048
       self.hidden_act = "silu"
       self.softmax_act = "softmax"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 32
       self.vocab_size = 32000
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

class llama7b_config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 32
       self.hidden_size = 4096
       self.intermediate_size = 11008
       self.max_position_embeddings = 2048
       self.hidden_act = "silu"
       self.softmax_act = "softmax"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 32
       self.vocab_size = 32000
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

def bench():
    # setup fake data for timing purpose
    commInit = crypten.communicator.get().get_communication_stats()
    input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float().cuda()

    timing = defaultdict(float)

    m = llama(config, timing)
    model = encrypt_model(m, llama, (config, timing), input_ids).eval()

    # encrpy inputs
    input_ids = encrypt_tensor(input_ids)

    for i in range(1):
        m.reset_timing()
        time_s = time.time()
        # run a forward pass
        with crypten.no_grad():
            model.generate(input_ids, config.new_tokens)

        time_e = time.time()
        timing["total_time"] = (time_e - time_s)
        print(timing)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-m", "--model-name", default="llama7b")
    parser.add_argument("-r", "--rank", default=0)
    parser.add_argument("-s", "--seq-len", default=512)
    parser.add_argument("-n", "--new-tokens", default=5)
    parser.add_argument("-H", "--head-compress", default=1)
    parser.add_argument("-p", "--head-pruning", default=0.0)
    parser.add_argument("-l", "--use-lora", default="False")
    parser.add_argument("-R", "--lora-rank", default=4)
    # parser.add_argument("-E", "--skip-embed", default="True")
    # parser.add_argument("-L", "--skip-lm", default="True")
    parser.add_argument("-c", "--cuda", default="True")
    parser.add_argument("-ga", "--gelu-approx", default="0")
    parser.add_argument("-sa", "--softmax-approx", default="0")
    args = parser.parse_args()

    rank = int(args.rank)
    model_name = args.model_name
    seq_len = int(args.seq_len)
    new_tokens = int(args.new_tokens)
    head_compress = int(args.head_compress)
    use_lora = args.use_lora == "True"
    lora_rank = int(args.lora_rank)
    # skip_embed = args.skip_embed == "True"
    # skip_lm = args.skip_lm == "True"
    use_cuda = args.cuda == "True"
    head_pruning = float(args.head_pruning)

    #'''
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["MASTER_ADDR"] = "10.128.0.41"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RENDEZVOUS"] = "env://"
    #'''
    cfg.mpc.provider = "TTP"
    cfg.communicator.verbose = True

    print(f"model name: {model_name}, seq len: {seq_len}, new tokens: {new_tokens}, head compress: {head_compress}, use lora: {use_lora}, lora rank: {lora_rank}, head_pruning: {head_pruning}")

    if model_name == "llama3b":
        config = llama3b_config()
    elif model_name == "llama7b":
        config = llama7b_config()
    else:
        raise NotImplementedError

    if head_pruning > 0.0:
        config.head_pruning = True
        config.num_attention_heads = int(config.num_attention_heads * (1 - head_pruning))
        config.pruned_hidden_size = int(config.hidden_size * (1 - head_pruning))
        assert(head_compress == 1)
    else:
        config.head_pruning = None

    if args.gelu_approx == "0":
        config.hidden_act = "silu"
    elif args.gelu_approx == "1":
        config.hidden_act = "quad"
    elif args.gelu_approx == "2":
        config.hidden_act = "relu"
    else:
        raise NotImplementedError
    
    if args.softmax_approx == "0":
        config.softmax_act = "softmax"
    elif args.softmax_approx == "1":
        config.softmax_act = "softmax_2QUAD"
    elif args.softmax_approx == "2":
        config.softmax_act = "softmax_L2QUAD"
    elif args.softmax_approx == "3":
        config.softmax_act = "softmax_2RELU"
    else:
        raise NotImplementedError

    config.num_hidden_layers = 1
    assert(config.num_attention_heads % head_compress == 0)
    config.num_attention_heads = config.num_attention_heads // head_compress
    config.tie_word_embeddings = False
    config.sequence_length = seq_len
    config.new_tokens = new_tokens
    config.lora = use_lora
    config.lora_dim = lora_rank

    print(f"using model config: {config}")

    crypten.init()
    if rank == 2:
        crypten.mpc.provider.TTPServer()
    else:
        bench()
