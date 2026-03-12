import argparse
import json

import jax.numpy as jnp
from jax import random
import spu.utils.distributed as ppd

import time, json
import spu.utils.simulation as pps
import spu
import math
from enum import Enum

copts = spu.spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

class OpType(Enum):
    MATMUL = 0
    COMPARISON = 1
    ELEMWISEPROD = 2

def bench(inp_1, inp_2):
    out = inp_2
    # for i in range(reps):
    #     out = jnp.matmul(inp_1, out)
    #'''
    for i in range(reps):
        if op_type == OpType.MATMUL:
            out = jnp.matmul(inp_1, out)
        elif op_type == OpType.COMPARISON:
            out = jnp.maximum(inp_1, out)
        elif op_type == OpType.ELEMWISEPROD:
            out = jnp.multiply(inp_1, out)
        else:
            raise NotImplementedError
    #'''
    return out

def run_on_spu():
    if not simulation:
        start = time.time()
        inp_1_ = ppd.device("P1")(lambda x: x)(inp_1)
        inp_2_ = ppd.device("P2")(lambda x: x)(inp_2)
        end = time.time()
        print(f"Elapsed Time (SPU Input): {end - start} seconds")

        start = time.time()
        out = ppd.device("SPU")(bench, copts=copts)(inp_1_, inp_2_)
        out = ppd.get(out)
        end = time.time()
        print(f"Elapsed Time (SPU Protocol): {end - start} seconds")
    else:
        start = time.time()
        spu_bench = pps.sim_jax(simulator, bench, copts=copts)
        out = spu_bench(inp_1, inp_2)
        end = time.time()
        print(f"Elapsed Time (SPU Simulation): {end - start} seconds")
    return


if __name__ == '__main__':
    global simulation, simulator, op_type, inp_1, inp_2, reps

    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-c", "--config", default="./3pc.json")
    parser.add_argument("-s1", "--size-1", default=100)
    parser.add_argument("-s2", "--size-2", default=100)
    parser.add_argument("-s3", "--size-3", default=100)
    parser.add_argument("-r", "--reps", default=10)
    parser.add_argument("-o", "--op", default="MATMUL")
    parser.add_argument("-S", "--simulation", default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    s1 = int(args.size_1)
    s2 = int(args.size_2)
    s3 = int(args.size_3)
    reps = int(args.reps)
    simulation = args.simulation == "True"

    key = random.PRNGKey(0)
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
    else:
        raise NotImplementedError

    inp_1 = random.normal(key, shape_1)
    inp_2 = random.normal(key, shape_2)

    if not simulation:
        ppd.init(conf["nodes"], conf["devices"])
    else:
        # config = conf["devices"]["SPU"]["config"]["runtime_config"]
        # protocol = spu.ProtocolKind.REF2K
        protocol = spu.ProtocolKind.CHEETAH
        num_parties = 2#1
        field = spu.FieldType.FM64
        sim_config = spu.RuntimeConfig(protocol=protocol, field=field)
        sim_config.enable_pphlo_profile = True
        sim_config.enable_hal_profile = True
        sim_config.fxp_exp_mode = 0
        sim_config.fxp_exp_iters = 5
        simulator = pps.Simulator(num_parties, sim_config)

    print(f"microbenchmark size: ({s1}, {s2}, {s3}), reps: {reps}, op type: {op_type}")
    print(f"running on simulation? {simulation}")

    run_on_spu()