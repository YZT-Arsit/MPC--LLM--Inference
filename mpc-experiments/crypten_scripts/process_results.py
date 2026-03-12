# 0: defaultdict(<class 'float'>, {'EmbedTime': 10.632102727890015, 'EmbedCommTime': 8.428680103999966, 'EmbedCommByte': 3578304512.0, 'NormTime': 1.2191333770751953, 'NormCommTime': 0.7996653390006827, 'NormCommByte': 357198696.0, 'LinearTime': 16.33236336708069, 'LinearCommTime': 11.570162457000947, 'LinearCommByte': 10523964368.0, 'SoftmaxTime': 18.907024383544922, 'SoftmaxCommTime': 16.064742234000505, 'SoftmaxCommByte': 14377897600.0, 'ActTime': 4.5427467823028564, 'ActCommTime': 4.450314823999975, 'ActCommByte': 4529855720.0, 'LayerTime': 33.1493079662323, 'lmHeadTime': 8.168398380279541, 'lmHeadCommTime': 5.791284419000249, 'lmHeadCommByte': 5124954112.0, 'GenerateMainTime': 52.83754229545593, 'GenerateMainCommTime': 42.12838555300101, 'GenerateMainCommByte': 33724419592.0, 'GenerateOtherTime': 1.7937581539154053, 'GenerateOtherCommTime': 0.9868131129975382, 'GenerateOtherCommByte': 99816000.0, 'cur_step_total': 54.631317138671875})
from ast import literal_eval
import re
import argparse
import enum
import os

class ActType(enum.Enum):
    Default = 0
    Quad = 1
    ReLU = 2

class SoftmaxType(enum.Enum):
    Default = 0
    TwoQuad = 1
    LearnableTwoQuad = 2
    TwoReLU = 3

ActTypeMap = {
    0: "SiLU",
    1: "Quad",
    2: "ReLU",
}

SoftmaxTypeMap = {
    0: "Softmax",
    1: "2Quad",
    2: "L2Quad",
    3: "2ReLU",
}

def get_dicts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Extracting the dictionary part from the string
        dicts_str = [re.search(r'\{.*\}', s).group() for i, s in enumerate(lines) if "defaultdict" in s]
        # Converting the string to a dictionary
        dicts = [literal_eval(s) for s in dicts_str]
        return dicts

def process_dicts(dict_0, dict_1):
    embed_time = max(dict_0["EmbedTime"], dict_1["EmbedTime"])
    embed_comm = dict_0["EmbedCommByte"] + dict_1["EmbedCommByte"]

    lmhead_time = max(dict_0["lmHeadTime"], dict_1["lmHeadTime"])
    lmhead_comm = dict_0["lmHeadCommByte"] + dict_1["lmHeadCommByte"]

    generate_other_time = max(dict_0["GenerateOtherTime"], dict_1["GenerateOtherTime"])
    generate_other_comm = dict_0["GenerateOtherCommByte"] + dict_1["GenerateOtherCommByte"]

    generate_main_time = max(dict_0["GenerateMainTime"], dict_1["GenerateMainTime"])
    generate_main_comm = dict_0["GenerateMainCommByte"] + dict_1["GenerateMainCommByte"]

    exp_layer_time = max(dict_0["LayerTime"], dict_1["LayerTime"])

    layer_time = generate_main_time - embed_time - lmhead_time
    layer_comm = generate_main_comm - embed_comm - lmhead_comm

    other_time = lmhead_time + generate_other_time
    other_comm = lmhead_comm + generate_other_comm

    return {
        'LayerTime': layer_time,
        'LayerComm': layer_comm / (1024 ** 3),
        'ExpLayerTime': exp_layer_time,
        'EmbedTime': embed_time,
        'EmbedComm': embed_comm / (1024 ** 3),
        'OtherTime': other_time,
        'OtherComm': other_comm / (1024 ** 3),
    }

def combine_dicts(prefilling_dict, decoding_dict):
    res = {}
    for key in prefilling_dict:
        res['Prefilling ' + key] = prefilling_dict[key]
        res['Decoding ' + key] = decoding_dict[key]
    return res

'''
def estimate_runtime(processed_dict, num_layers, frozen):
    layer_time = processed_dict['LayerTime']
    layer_comm = processed_dict['LayerComm']
    embed_time = processed_dict['EmbedTime']
    embed_comm = processed_dict['EmbedComm']
    other_time = processed_dict['OtherTime']
    other_comm = processed_dict['OtherComm']

    return {
        'TotalTime': round(layer_time * num_layers + embed_time + other_time, 2),
        'TotalComm': round(layer_comm * num_layers + embed_comm + other_comm, 2),
        'TotalTimeWithLF': round(layer_time * (num_layers - frozen) + other_time, 2),
        'TotalCommWithLF': round(layer_comm * (num_layers - frozen) + other_comm, 2),
    }
'''

def get_final_dict(filenames):
    dicts_0 = get_dicts(filenames[0])
    dicts_1 = get_dicts(filenames[1])

    prefilling_dict = process_dicts(dicts_0[0], dicts_1[0])
    decoding_dict = process_dicts(dicts_0[1], dicts_1[1])
    combined_dict = combine_dicts(prefilling_dict, decoding_dict)
    return combined_dict

def get_filename(seq_len, head_compress, use_lora, lora_rank, head_pruning=0.0, gelu_approx=0, softmax_approx=0):
    # Get the full path to the current file
    module_path = __file__
    # Extract the directory where the current file is located
    module_dir = os.path.dirname(module_path)

    prefix = f"{module_dir}/../results/crypten/mllama3b-s{seq_len}-H{head_compress}-l{use_lora}-R{lora_rank}{f'-p{head_pruning}' if head_pruning > 0.0 else ''}-n5-ga{gelu_approx}-sa{softmax_approx}"
    return f"{prefix}-r0-crypten.log", f"{prefix}-r1-crypten.log"

def add_improvement(results, baseline_results):
    for metric in baseline_results:
        results[metric + ' Improvement'] = round(baseline_results[metric] / results[metric], 1)
        # del results[metric]
    return results

get_stats = lambda stat, metric, stage, w_frozen: round(stat[f'{stage} Layer{metric}'] * (total_layers - w_frozen * frozen) + (1 - w_frozen) * stat[f'{stage} Embed{metric}'] + stat[f'{stage} Other{metric}'], 2)
get_stats_dict = lambda stat, w_frozen: {'Prefilling Time': get_stats(stat, 'Time', 'Prefilling', w_frozen), 'Prefilling Comm': get_stats(stat, 'Comm', 'Prefilling', w_frozen), 'Decoding Time': get_stats(stat, 'Time', 'Decoding', w_frozen), 'Decoding Comm': get_stats(stat, 'Comm', 'Decoding', w_frozen)}

def process_results(baseline_results, stats, w_frozen):
    results = get_stats_dict(stats, w_frozen)
    results = add_improvement(results, baseline_results)
    return results 

def get_result(seq_len, frozen_, total_layers_, head_compress, use_lora, lora_rank, gelu_approx=0, softmax_approx=0):
    global frozen, total_layers
    frozen = frozen_
    total_layers = total_layers_

    std_stats = get_final_dict(get_filename(seq_len, 1, False, lora_rank))
    baseline_results = get_stats_dict(std_stats, False)
    stats = get_final_dict(get_filename(seq_len, head_compress, use_lora, lora_rank, gelu_approx, softmax_approx))
    results = process_results(baseline_results, stats, frozen is not None)
    return results

def crypten_results_processor(seq_len, head_compress, lora_rank, frozen_, total_layers_, verbose=False):
    global frozen, total_layers
    frozen = frozen_
    total_layers = total_layers_

    std_filenames = get_filename(seq_len, 1, False, lora_rank)
    std_stats = get_final_dict(std_filenames)
    hc_filenames = get_filename(seq_len, head_compress, False, lora_rank)
    hc_stats = get_final_dict(hc_filenames)
    hp_filenames = get_filename(seq_len, 1, False, lora_rank, 0.75)
    hp_stats = get_final_dict(hp_filenames)
    lora_filenames = get_filename(seq_len, 1, True, lora_rank)
    lora_stats = get_final_dict(lora_filenames)
    hc_lora_filenames = get_filename(seq_len, head_compress, True, lora_rank)
    hc_lora_stats = get_final_dict(hc_lora_filenames)

    baseline_results = get_stats_dict(std_stats, False)
    hc_results = process_results(baseline_results, hc_stats, False)
    hp_results = process_results(baseline_results, hp_stats, False)
    lora_results = process_results(baseline_results, lora_stats, False)
    lf_results = process_results(baseline_results, std_stats, True)
    lf_hc_results = process_results(baseline_results, hc_stats, True)
    lf_hp_results = process_results(baseline_results, hp_stats, True)
    lf_lora_results = process_results(baseline_results, lora_stats, True)
    lf_hc_lora_results = process_results(baseline_results, hc_lora_stats, True)

    if verbose:
        print("--------------------\nBaseline Results\n--------------------")
        print(baseline_results)

        print("--------------------\nHC Results\n--------------------")
        print(hc_results)

        print("--------------------\nHP Results\n--------------------")
        print(hp_results)

        print("--------------------\nLoRA Results\n--------------------")
        print(lora_results)

        print("--------------------\nLF Results\n--------------------")
        print(lf_results)

        print("--------------------\nLF + HC Results\n--------------------")
        print(lf_hc_results)

        print("--------------------\nLF + HP Results\n--------------------")
        print(lf_hp_results)

        print("--------------------\nLF + LoRA Results\n--------------------")
        print(lf_lora_results)

        print("--------------------\nLF + HC + LoRA Results\n--------------------")
        print(lf_hc_lora_results)

    mpcformer_results = []
    for gelu_approx in [0, 2, 1]:#[0, 1, 2]:
        for softmax_approx in [0, 3, 2, 1]:#[0, 1, 2, 3]:
            if gelu_approx == 0 and softmax_approx == 0:
                continue
            filenames = get_filename(seq_len=seq_len, head_compress=1, use_lora=False, lora_rank=lora_rank, head_pruning=0.0, gelu_approx=gelu_approx, softmax_approx=softmax_approx)
            stats = get_final_dict(filenames)
            results = process_results(baseline_results, stats, False)
            if verbose:
                print(f"--------------------\nMPCFormer (Activation: {ActTypeMap[gelu_approx]}, Softmax: {SoftmaxTypeMap[softmax_approx]}) Results\n--------------------")
                print(results)
            mpcformer_results.append((ActTypeMap[gelu_approx], SoftmaxTypeMap[softmax_approx], results))

    MARILL_MAP = {
        f"LF={frozen}": lf_results,
        f"HC={head_compress}": hc_results,
        f"HP=0.75": hp_results,
        f"LoRA={lora_rank}": lora_results,
        "LF+HC": lf_hc_results,
        "LF+HP": lf_hp_results,
        "LF+LoRA": lf_lora_results,
    }
    MPCFORMER_MAP = {}
    for result in mpcformer_results:
        '''
        print_gelu_approx = result[0] if result[0] is not None else ""
        print_plus = "+" if result[0] is not None and result[1] is not None else ""
        print_softmax_approx = result[1] if result[1] is not None else ""
        MPCFORMER_MAP[f"{print_gelu_approx}{print_plus}{print_softmax_approx}"] = result[2]
        '''
        MPCFORMER_MAP[f"{result[0]}+{result[1]}"] = result[2]

    return (MARILL_MAP, MPCFORMER_MAP)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-s", "--seq-len", default=2040)
    parser.add_argument("-H", "--head-compress", default=4)
    parser.add_argument("-l", "--use-lora", default="False")
    parser.add_argument("-R", "--lora-rank", default=64)
    parser.add_argument("-f", "--frozen", default=13)
    args = parser.parse_args()

    seq_len = int(args.seq_len)
    head_compress = int(args.head_compress)
    use_lora = args.use_lora == "True"
    lora_rank = int(args.lora_rank)
    frozen = int(args.frozen)

    total_layers = 26

    crypten_results_processor(seq_len, head_compress, lora_rank, frozen, total_layers, verbose=True)
