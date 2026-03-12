import subprocess

def read_file_from_index(file_path, start_index=0):
    try:
        with open(file_path, 'r') as file:
            # Move the file pointer to the start_index
            file.seek(start_index)
            # Read the file contents from start_index
            file_contents = file.read()
            # Current file pointer position after reading
            current_index = file.tell()
            return file_contents, current_index

    except FileNotFoundError:
        return "File not found.", None
    except IOError:
        return "An error occurred while reading the file.", None
    except ValueError:
        return "Invalid input for file index.", None

def exec_cmd(model_name, config, seq_len, head_compress, use_lora, lora_rank, head_pruning, skip_embed, skip_lm, gelu_approx=0, softmax_approx=0):
    global log_index

    cp_string_before = f"################################"
    cp_string = f"model_name: {model_name}, config: {config}, seq_len: {seq_len}, head_compression: {head_compress}, use_lora: {use_lora}, lora_rank: {lora_rank}, head_pruning: {head_pruning}, skip_embed: {skip_embed}, skip_lm_head: {skip_lm}, gelu_approx: {gelu_approx}, softmax_approx: {softmax_approx}"
    cp_string_after = f"################################"
    cp_string = cp_string_before + "\n" + cp_string + "\n" + cp_string_after + "\n"
    print(cp_string)

    exp_filename = f"{model_name}-{config}-s{seq_len}-H{head_compress}-l{use_lora}-r{lora_rank}{f'-p{head_pruning}' if head_pruning > 0.0 else ''}-E{skip_embed}-L{skip_lm}-ga{gelu_approx}-sa{softmax_approx}-spu.log"
    with open(exp_filename, 'a') as file:
        file.write(cp_string)

    # cmd = f"python bench.py -m {model_name} -c {config}.json -s {seq_len} -n 10 -H {head_compress} -l {use_lora} -r {lora_rank} -E {skip_embed} -L {skip_lm} -S False  2>&1 | tee {model_name}-{config}-s{seq_len}-H{head_compress}-l{use_lora}-r{lora_rank}-E{skip_embed}-L{skip_lm}.log"
    cmd = f"python bench.py -m {model_name} -c {config}.json -s {seq_len} -n 10 -H {head_compress} -l {use_lora} -r {lora_rank} -p {head_pruning} -E {skip_embed} -L {skip_lm} -S False -ga {gelu_approx} -sa {softmax_approx}"
    print(f"Executing {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    log_name = f"{config}-node:0.log"
    contents, index = read_file_from_index(log_name, log_index)
    with open(exp_filename, 'a') as file:
        file.write(contents)
    log_index = index

if __name__ == '__main__':
    lora_rank = 64

    # 3PC experiments
    log_index = 0
    config = "3pc"
    for model_name in ["llama3b"]:
        for seq_len in [2036]:
            '''
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="False", skip_lm="False", gelu_approx=0, softmax_approx=0)
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="False", skip_lm="True", gelu_approx=0, softmax_approx=0)
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="False", gelu_approx=0, softmax_approx=0)
            for gelu_approx in [0, 1, 2]:
                for softmax_approx in [0, 1, 2, 3]:
                    exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=gelu_approx, softmax_approx=softmax_approx)
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=4, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)
            for lora_rank in [64]:
                exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="True", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)
                exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=4, use_lora="True", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)
            '''
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.75, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)

    # 2PC experiments
    log_index = 0
    config = "2pc"
    for model_name in ["llama3b"]:
        for seq_len in [64]:
            '''
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="False", skip_lm="False", gelu_approx=0, softmax_approx=0)
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="False", skip_lm="True", gelu_approx=0, softmax_approx=0)
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="False", gelu_approx=0, softmax_approx=0)
            for gelu_approx in [0, 1, 2]:
                for softmax_approx in [0, 1, 2, 3]:
                    exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=gelu_approx, softmax_approx=softmax_approx)
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=4, use_lora="False", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)
            for lora_rank in [64]:
                exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="True", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)
                exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=4, use_lora="True", lora_rank=lora_rank, head_pruning=0.0, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)
            '''
            exec_cmd(model_name=model_name, config=config, seq_len=seq_len, head_compress=1, use_lora="False", lora_rank=lora_rank, head_pruning=0.75, skip_embed="True", skip_lm="True", gelu_approx=0, softmax_approx=0)