import argparse
import re
import enum
import matplotlib.pyplot as plt
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
        runtimes = [float(re.search(r'[0-9]*\.?[0-9]+$', s).group()) for i, s in enumerate(lines) if "HLO profiling: total time" in s]
        comms = [int(re.search(r'[0-9]*,', s).group()[:-1]) for i, s in enumerate(lines) if "Link details: total send bytes" in s]
        rounds = [int(re.search(r'[0-9]*$', s).group()) for i, s in enumerate(lines) if "Link details: total send bytes" in s]
        return {
            'Prefilling Time': runtimes[0],
            'Prefilling Comm': comms[0] / (1024 ** 3),
            'Prefilling Rounds': rounds[0],
            'Decoding Time': runtimes[1] / 10.0,
            'Decoding Comm': comms[1] / ((1024 ** 3) * 10.0),
            'Decoding Rounds': rounds[1] / 10.0,
        }

def get_filename(seq_len, head_compress, use_lora, lora_rank, setting, skip_embed, skip_lm, head_pruning=0.0, gelu_approx=0, softmax_approx=0):
    # Get the full path to the current file
    module_path = __file__
    # Extract the directory where the current file is located
    module_dir = os.path.dirname(module_path)

    prefix = f"{module_dir}/../results/spu/llama3b-{setting}-s{seq_len}-H{head_compress}-l{use_lora}-r{lora_rank}{f'-p{head_pruning}' if head_pruning > 0.0 else ''}-E{skip_embed}-L{skip_lm}-ga{gelu_approx}-sa{softmax_approx}-spu.log"
    return f"{prefix}"

def add_improvement(results, baseline_results):
    for metric in baseline_results:
        results[metric + ' Improvement'] = round(baseline_results[metric] / results[metric], 1)
        # del results[metric]
    return results

def embed_and_other(std_stats_w_embed, std_stats_w_lm, std_stats):
    embed = {}
    embed['Prefilling Time'] = max(std_stats_w_embed['Prefilling Time'] - std_stats['Prefilling Time'], 0)
    embed['Prefilling Comm'] = std_stats_w_embed['Prefilling Comm'] - std_stats['Prefilling Comm']
    embed['Prefilling Rounds'] = std_stats_w_embed['Prefilling Rounds'] - std_stats['Prefilling Rounds']
    embed['Decoding Time'] = max(std_stats_w_embed['Decoding Time'] - std_stats['Decoding Time'], 0)
    embed['Decoding Comm'] = std_stats_w_embed['Decoding Comm'] - std_stats['Decoding Comm']
    embed['Decoding Rounds'] = std_stats_w_embed['Decoding Rounds'] - std_stats['Decoding Rounds']
    # print("Embed Prefilling Time", embed['Prefilling Time'])
    # print("Embed Prefilling Comm", embed['Prefilling Comm'])
    # print("Embed Prefilling Rounds", embed['Prefilling Rounds'])
    # print("Embed Decoding Time", embed['Decoding Time'])
    # print("Embed Decoding Comm", embed['Decoding Comm'])

    other = {}
    other['Prefilling Time'] = max(std_stats_w_lm['Prefilling Time'] - std_stats['Prefilling Time'], 0)
    other['Prefilling Comm'] = std_stats_w_lm['Prefilling Comm'] - std_stats['Prefilling Comm']
    other['Prefilling Rounds'] = std_stats_w_lm['Prefilling Rounds'] - std_stats['Prefilling Rounds']
    other['Decoding Time'] = max(std_stats_w_lm['Decoding Time'] - std_stats['Decoding Time'], 0)
    other['Decoding Comm'] = std_stats_w_lm['Decoding Comm'] - std_stats['Decoding Comm']
    other['Decoding Rounds'] = std_stats_w_lm['Decoding Comm'] - std_stats['Decoding Rounds']
    # print("Other Prefilling Time", other['Prefilling Time'])
    # print("Other Prefilling Comm", other['Prefilling Comm'])
    # print("Other Prefilling Rounds", other['Prefilling Rounds'])
    # print("Other Decoding Time", other['Decoding Time'])
    # print("Other Decoding Comm", other['Decoding Comm'])

    return (embed, other)

get_stats_dict = lambda fn: {'Prefilling Time': fn('Prefilling Time'), 'Prefilling Comm': fn('Prefilling Comm'), 'Prefilling Rounds': fn('Prefilling Rounds'), 'Decoding Time': fn('Decoding Time'), 'Decoding Comm': fn('Decoding Comm'), 'Decoding Rounds': fn('Decoding Rounds')}

def process_results(stats, embed, other, total_layers, frozen=None, baseline=None):
    fn = lambda metric: round((frozen is None) * embed[metric] + (stats[metric] * (total_layers - (0 if frozen is None else frozen))) + other[metric], 2)
    results = get_stats_dict(fn)
    if baseline is not None:
        results = add_improvement(results, baseline)
    return results

def get_result(seq_len, frozen, total_layers, head_compress, use_lora, lora_rank, setting, gelu_approx=0, softmax_approx=0):
    std_stats_w_embed = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, False, True, 0.0, 0, 0))
    std_stats_w_lm = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, True, False, 0.0, 0, 0))
    std_stats = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, True, True, 0.0, 0, 0))
    stats = get_dicts(get_filename(seq_len, head_compress, use_lora, lora_rank, setting, True, True, 0.0, gelu_approx, softmax_approx))
    (embed, other) = embed_and_other(std_stats_w_embed, std_stats_w_lm, std_stats)
    baseline_results = process_results(std_stats, embed, other, total_layers, None, None)
    results = process_results(stats, embed, other, total_layers, frozen, baseline_results)
    return results

def spu_results_processor(seq_len, head_compress, lora_rank, frozen, total_layers, setting, verbose=False):

    # std_stats_w_embed_lm = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, False, False, 0, 0))
    std_stats_w_embed = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, False, True, 0.0, 0, 0))
    std_stats_w_lm = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, True, False, 0.0, 0, 0))
    std_stats = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, True, True, 0.0, 0, 0))
    hc_stats = get_dicts(get_filename(seq_len, head_compress, False, lora_rank, setting, True, True, 0.0, 0, 0))
    hp_stats = get_dicts(get_filename(seq_len, 1, False, lora_rank, setting, True, True, 0.75, 0, 0))
    lora_stats = get_dicts(get_filename(seq_len, 1, True, lora_rank, setting, True, True, 0.0, 0, 0))
    hc_lora_stats = get_dicts(get_filename(seq_len, head_compress, True, lora_rank, setting, True, True, 0.0, 0, 0))

    # process_results(std_stats_w_embed_lm, std_stats_w_embed, std_stats_w_lm, std_stats, hc_stats, lora_stats, hc_lora_stats, total_layers, frozen)
    (embed, other) = embed_and_other(std_stats_w_embed, std_stats_w_lm, std_stats)
    baseline_results = process_results(std_stats, embed, other, total_layers, None, None)
    hc_results = process_results(hc_stats, embed, other, total_layers, None, baseline_results)
    hp_results = process_results(hp_stats, embed, other, total_layers, None, baseline_results)
    lora_results = process_results(lora_stats, embed, other, total_layers, None, baseline_results)
    lf_results = process_results(std_stats, embed, other, total_layers, frozen, baseline_results)
    lf_hc_results = process_results(hc_stats, embed, other, total_layers, frozen, baseline_results)
    lf_hp_results = process_results(hp_stats, embed, other, total_layers, frozen, baseline_results)
    lf_lora_results = process_results(lora_stats, embed, other, total_layers, frozen, baseline_results)
    lf_hc_lora_results = process_results(hc_lora_stats, embed, other, total_layers, frozen, baseline_results)

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
        for softmax_approx in [0, 3, 2, 1]:# [0, 1, 2, 3]:
            if gelu_approx == 0 and softmax_approx == 0:
                continue
            filenames = get_filename(seq_len=seq_len, head_compress=1, use_lora=False, lora_rank=lora_rank, setting=setting, skip_embed=True, skip_lm=True, head_pruning=0.0, gelu_approx=gelu_approx, softmax_approx=softmax_approx)
            stats = get_dicts(filenames)
            results = process_results(stats, embed, other, total_layers, None, baseline_results)
            if verbose:
                print(f"--------------------\nMPCFormer (Activation: {ActTypeMap[gelu_approx]}, Softmax: {SoftmaxTypeMap[softmax_approx]}) Results\n--------------------")
                print(results)
            # mpcformer_results.append((ActTypeMap[gelu_approx] if gelu_approx != 0 else None, SoftmaxTypeMap[softmax_approx] if softmax_approx != 0 else None, results))
            mpcformer_results.append((ActTypeMap[gelu_approx], SoftmaxTypeMap[softmax_approx], results))

    '''
    ticks = [round((n-1)/float(n) * total_layers) for n in range(1, 8)]
    frozen_ablation = []
    for frozen in ticks:
        results = process_results(std_stats, embed, other, total_layers, frozen, baseline_results)
        print(f"--------------------\nLF (frozen={frozen}) Results\n--------------------")
        print(results)
        frozen_ablation.append(results)

    plt.figure(figsize=(10, 6))
    metrics = ['Prefilling Time Improvement', 'Prefilling Comm Improvement', 'Decoding Time Improvement', 'Decoding Comm Improvement']
    for metric in metrics:
        metric_values = [d[metric] for d in frozen_ablation]
        plt.plot(ticks, metric_values, label=metric)
    plt.plot(ticks, [total_layers/(total_layers-f) for f in ticks], label="Claimed Improvement (fx)")

    # Adding titles and labels
    plt.title('Improvement vs. #Frozen Layers')
    plt.xlabel('#Frozen Layers')
    plt.ylabel('Improvement over Baseline')
    plt.xticks(ticks)  # Ensure we have a tick for every frozen value
    plt.legend()

    # Display the plot
    plt.show()
    #'''

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
    parser.add_argument("-s", "--seq-len", default=501)
    parser.add_argument("-H", "--head-compress", default=4)
    # parser.add_argument("-l", "--use-lora", default="False")
    parser.add_argument("-R", "--lora-rank", default=64)
    parser.add_argument("-f", "--frozen", default=13)
    parser.add_argument("-c", "--config", default="3pc")
    args = parser.parse_args()

    seq_len = int(args.seq_len)
    head_compress = int(args.head_compress)
    lora_rank = int(args.lora_rank)
    frozen = int(args.frozen)
    setting = args.config

    if setting == "3pc":
        seq_len = 2036
    elif setting == "2pc":
        seq_len = 64
    else:
        raise ValueError(f"Invalid config {setting}")

    spu_results_processor(seq_len, head_compress, lora_rank, frozen, 26, setting, verbose=True)

    '''
    metrics = ['Prefilling Time Improvement', 'Prefilling Comm Improvement', 'Decoding Time Improvement', 'Decoding Comm Improvement']

    for metric in metrics[:1]:
        bar_labels_marill = list(MARILL_MAP.keys())
        bar_labels_mpcformer = list(MPCFORMER_MAP.keys())
        bar_values_marill = [MARILL_MAP[key][metric] for key in bar_labels_marill]
        bar_values_mpcformer = [MPCFORMER_MAP[key][metric] for key in bar_labels_mpcformer]
        
        bar_labels = bar_labels_marill + bar_labels_mpcformer
        bar_values = bar_values_marill + bar_values_mpcformer
        max_height = max(bar_values)

        cluster_labels = ["$\\textsc{Marill}$", "MPC-friendly Approximations"]
        cluster_positions = [len(bar_labels_marill) // 2, len(bar_labels_marill) + 1 + len(bar_labels_mpcformer) // 2]

        # Create gap between clusters
        bar_positions = list(range(len(bar_labels_marill))) + list(range(len(bar_labels_marill) + 1, len(bar_labels) + 1))

        fig, ax = plt.subplots()
        bars = plt.bar(bar_positions, bar_values)

        # Adding individual bar labels
        plt.xticks(bar_positions, bar_labels, rotation=60, ha="right")

        # Annotate clusters
        for label, pos in zip(cluster_labels, cluster_positions):
            plt.text(pos, max_height + 1, label, ha='center', va='bottom')

        # Adding a label above each bar with the improvement factor succeeded by "x"
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}$\\times$' if height != 1.0 else '1$\\times$',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Increase height to ensure annotation fits in the plot
        plt.ylim(top=max_height + 2)
        # Adding titles and labels
        plt.title(f"{setting.upper()} - {metric}")
        # plt.xlabel('Variant')
        plt.ylabel('Improvement over Baseline')
        # plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        '''