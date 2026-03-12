import crypten
from crypten import mpc
import torch
from enum import Enum
import argparse
import time
from crypten.config import cfg
import os

# needed on MacOS to avoid serialization issues with spawn
import multiprocessing
multiprocessing.set_start_method('fork')

class OpType(Enum):
    MATMUL = 0
    COMPARISON = 1
    ELEMWISEPROD = 2
    DOTPROD = 3

# @mpc.run_multiprocess(world_size=2)
def microbench():
    print(f"world size: {crypten.communicator.get().get_world_size()}, ttp required: {crypten.mpc.ttp_required()}")
    inp_1_enc = crypten.cryptensor(inp_1)
    inp_2_enc = crypten.cryptensor(inp_2)
    # print(f"Communication Stats (Init): {crypten.communicator.get().get_communication_stats()}")
    start = time.time()
    for i in range(reps):
        if op_type == OpType.MATMUL:
            out = inp_1_enc.matmul(inp_2_enc)
        elif op_type == OpType.COMPARISON:
            out = inp_1_enc > inp_2_enc
        elif op_type == OpType.ELEMWISEPROD:
            out = inp_1_enc * inp_2_enc
        elif op_type == OpType.DOTPROD:
            out = inp_1_enc.matmul(inp_2_enc)
        else:
            raise NotImplementedError
    end = time.time()
    print(f"Elapsed Time (Crypten): {end - start} seconds")
    print(f"Communication Stats (Final): {crypten.communicator.get().get_communication_stats()}")
    return out

if __name__ == '__main__':
    global op_type, inp_1, inp_2, reps, shape_1, shape_2

    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-s1", "--size-1", default=100)
    parser.add_argument("-s2", "--size-2", default=100)
    parser.add_argument("-s3", "--size-3", default=100)
    parser.add_argument("-R", "--reps", default=100)
    parser.add_argument("-o", "--op", default="MATMUL")
    parser.add_argument("-r", "--rank", default=0)
    parser.add_argument("-c", "--cuda", default="True")
    args = parser.parse_args()

    s1 = int(args.size_1)
    s2 = int(args.size_2)
    s3 = int(args.size_3)
    reps = int(args.reps)
    rank = int(args.rank)
    use_cuda = args.cuda == "True"

    #'''
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["MASTER_ADDR"] = "10.128.0.41"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RENDEZVOUS"] = "env://"
    #'''
    cfg.mpc.provider = "TTP"
    cfg.communicator.verbose = True

    # crypten.init()
    # crypten.mpc.set_default_provider("TTP")

    if args.op == "MATMUL":
        op_type = OpType.MATMUL
        shape_1 = (s1, s2)
        shape_2 = (s2, s3)
    elif args.op == "COMPARISON":
        op_type = OpType.COMPARISON
        shape_1 = (s1 * s2 * s3, )
        shape_2 = shape_1
    elif args.op == "ELEMWISE":
        op_type = OpType.ELEMWISEPROD
        shape_1 = (s1 * s2 * s3, )
        shape_2 = shape_1
    elif args.op == "DOTPROD":
        op_type = OpType.DOTPROD
        shape_1 = (1, s1 * s2 * s3)
        shape_2 = (s1 * s2 * s3, 1)
    else:
        raise NotImplementedError

    inp_1 = torch.rand(shape_1)
    inp_2 = torch.rand(shape_2)
    if use_cuda:
        inp_1 = inp_1.cuda()
        inp_2 = inp_2.cuda()
    # inp_1_enc = crypten.cryptensor(inp_1)
    # inp_2_enc = crypten.cryptensor(inp_2)

    print(f"microbenchmark size: ({s1}, {s2}, {s3}), reps: {reps}, op type: {op_type}, rank: {rank}, using cuda: {use_cuda}")

    crypten.init()
    if rank == 2:
        crypten.mpc.provider.TTPServer()
    else:
        microbench()