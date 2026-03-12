import subprocess
import sys

def exec_cmd(model_name, rank, seq_len, head_compress, use_lora, lora_rank, head_pruning, gelu_approx=0, softmax_approx=0):

    exp_filename = f"m{model_name}-s{seq_len}-H{head_compress}-l{use_lora}-R{lora_rank}{f'-p{head_pruning}' if head_pruning > 0.0 else ''}-n5-ga{gelu_approx}-sa{softmax_approx}-r{rank}-crypten.log"

    cmd = f"python bench.py -m {model_name} -r {rank} -s {seq_len} -n 5 -H {head_compress} -l {use_lora} -R {lora_rank} -p {head_pruning} -ga {gelu_approx} -sa {softmax_approx} 2>&1 | tee {exp_filename}"
    print(f"Executing {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == '__main__':
    rank = int(sys.argv[1])
    lora_rank = 64
    for model_name in ["llama3b"]:
        for seq_len in [2040]:
            '''
            for gelu_approx in [0, 1, 2]:
                for softmax_approx in [0, 1, 2, 3]:
                    exec_cmd(model_name=model_name, rank=rank, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, gelu_approx=gelu_approx, softmax_approx=softmax_approx)
            exec_cmd(model_name=model_name, rank=rank, seq_len=seq_len, head_compress=4, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, gelu_approx=0, softmax_approx=0)
            exec_cmd(model_name=model_name, rank=rank, seq_len=seq_len, head_compress=1, use_lora="True", lora_rank=lora_rank, head_pruning=0.0, gelu_approx=0, softmax_approx=0)
            exec_cmd(model_name=model_name, rank=rank, seq_len=seq_len, head_compress=4, use_lora="True", lora_rank=lora_rank, head_pruning=0.0, gelu_approx=0, softmax_approx=0)
            '''
            exec_cmd(model_name=model_name, rank=rank, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.75, gelu_approx=0, softmax_approx=0)