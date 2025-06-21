import os
import json
import matplotlib.pyplot as plt
import numpy as np



results_easy = [
    {
        "exp_name": "Ask GPT-4o-mini for everything",
        "source": "external API",
        "acc_ground_truth": 0.8912,
        "acc_generations": 1.0,
        "latency": 1465,
        "marker": "s",
        "color": "blue",
        "alpha": 0.5,
        "Parameters": None,
        "Cost": 6.889701300000014
    },
    {
        "exp_name": "ScaleLLM small model",
        "source": "external API",
        "acc_ground_truth": 0.89,
        "acc_generations": 0.9209,
        "latency": 34.198 + 2.038 + (1024 / 117224) * 1465,
        "train_samples": 1024, 
        "marker": "o",
        "color": "blue",
        "alpha": 1.0,
        "Parameters": "347k",
        "Cost": 0.0588
    },
    # {
    #     "exp_name": "Small Yahoo 347k params 1024 train samples"
    #     "source": "external API",
    #     "acc_ground_truth": 0.89,
    #     "acc_generations": 0.9209,
    #     "latency": 34.198 + 2.038 + 491.604 + (1024 / 117224) * 1465,
    #     "train_samples": 1024,
    #     "marker": "*",
    #     "color": "blue",
    #     "alpha": 1.0,
    #     "Parameters": "347k"
    # },
    {
        "exp_name": "Ask Llama 3.3 70B for everything",
        "source": "local Llama 3.3 70B",
        "acc_ground_truth": 0.8888,
        "acc_generations": 1.0,
        "latency": 765032.137698,
        "marker": "s",
        "color": "green",
        "alpha": 1.0,
        "Parameters": "70B",
        "Cost": None
    },
    {
        "exp_name": "ScaleLLM small model",
        "source": "local Llama 3.3 70B",
        "acc_ground_truth": 0.89,
        "acc_generations": 0.9209,
        "latency": 34.198 + 2.038 + 6682,
        "train_samples": 1024, 
        "marker": "o",
        "color": "green",
        "alpha": 1.0,
        "Parameters": "347k",
        "Cost": None
    },
    # {
    #     "exp_name": "Small Yahoo 347k params 1024 train samples with embeddings (trained on local Llama 3.3 70B)",
    #     "source": "local Llama 3.3 70B",
    #     "acc_ground_truth": 0.89,
    #     "acc_generations": 0.9209,
    #     "latency": 34.198 + 2.038 + 491.604 + 6682,
    #     "train_samples": 1024,
    #     "marker": "*",
    #     "color": "green",
    #     "alpha": 1.0,
    # },

    {
        "exp_name": "SmolLM2-135M-Instruct finetuned",
        "source": "external API",
        "acc_ground_truth": 0.80859375,
        "acc_generations": 0.73046875,
        "latency": 12084 + 88.689 + (1024 / 117224) * 1465,
        "marker": "x",
        "color": "blue",
        "alpha": 1.0,
        "Parameters": "135M",
        "Cost": None
    },
    {
        "exp_name": "SmolLM2-135M-Instruct finetuned",
        "source": "local Llama 3.3 70B",
        "acc_ground_truth": 0.80859375,
        "acc_generations": 0.73046875,
        "latency": 12084 + 88.689 + 6682,
        "marker": "x",
        "color": "green",
        "alpha": 1.0,
        "Parameters": "135M",
        "Cost": None
    },
    {
        "exp_name": "SmolLM2-135M-Instruct",
        "source": None,
        "acc_ground_truth": 0.8037109375,
        "acc_generations": 0.7275390625,
        "latency": 12084,
        "marker": "x",
        "color": "orange",
        "alpha": 1.0,
        "Parameters": "135M",
        "Cost": None
    },
]


results_multi_tags = [
    {
        "exp_name": "Ask GPT-4o-mini for everything",
        "source": "external API",
        "f1_score_ground_truth": 0.450007706169678,
        "f1_score_generation": None,
        "latency": 1502,
        "marker": "s",
        "color": "blue",
        "alpha": 0.5,
        "Parameters": None,
        "Cost": 8.51925164999996
    },
    {
        "exp_name": "Small Model 347k params w/ 1024 train samples",
        "source": "external API",
        "f1_score_ground_truth": 0.43499908419317423,
        "f1_score_generation": None,
        # inference + training + collect samples
        "latency": 74.79454207420349 + 38 + (1024 / 117224) * 1502,
        "train_samples": 1024, 
        "marker": "o",
        "color": "blue",
        "alpha": 1.0,
        "Parameters": "347k",
        "Cost": 0.0588
    },
    # {
    #     "exp_name": "Small Yahoo 347k params 1024 train samples"
    #     "source": "external API",
    #     "acc_ground_truth": 0.89,
    #     "acc_generations": 0.9209,
    #     "latency": 34.198 + 2.038 + 491.604 + (1024 / 117224) * 1465,
    #     "train_samples": 1024,
    #     "marker": "*",
    #     "color": "blue",
    #     "alpha": 1.0,
    #     "Parameters": "347k"
    # },
    {
        "exp_name": "Ask Llama 3.3 70B for everything",
        "source": "local Llama 3.3 70B",
        "f1_score_ground_truth": 0.4390202124015074,
        "f1_score_generation": None,
        "latency": 988798,
        "marker": "s",
        "color": "green",
        "alpha": 1.0,
        "Parameters": "70B",
        "Cost": None
    },
    {
        "exp_name": "Small Model 4M params w/ 1024 train samples",
        "source": "local Llama 3.3 70B",
        "f1_score_ground_truth": 0.43133517233794916,
        "f1_score_generation": None,
        # 6682 for 1024 samples
        "latency": 107 + 36.24 + 8115,
        "train_samples": 1024, 
        "marker": "o",
        "color": "green",
        "alpha": 1.0,
        "Parameters": "4M",
        "Cost": None
    },
    # {
    #     "exp_name": "Small Yahoo 347k params 1024 train samples with embeddings (trained on local Llama 3.3 70B)",
    #     "source": "local Llama 3.3 70B",
    #     "acc_ground_truth": 0.89,
    #     "acc_generations": 0.9209,
    #     "latency": 34.198 + 2.038 + 491.604 + 6682,
    #     "train_samples": 1024,
    #     "marker": "*",
    #     "color": "green",
    #     "alpha": 1.0,
    # },
    
    {
        "exp_name": "SmolLM2-135M-Instruct finetuned w/ 1024 train samples",
        "source": "external API",
        "f1_score_ground_truth": 0.1420269312544295,
        "f1_score_generation": 0.448476821192053,
        # inference + training + collect samples
        "latency": 35354 + 317 + 8115,
        "marker": "x",
        "color": "blue",
        "alpha": 1.0,
        "Parameters": "135M",
        "Cost": None
    },
    {
        "exp_name": "SmolLM2-135M-Instruct finetuned w/ 1024 train samples",
        "source": "local Llama 3.3 70B",
        "f1_score_ground_truth": 0.12863933452168747,
        "f1_score_generation": 0.448476821192053,
        "latency": 35354 + 318 + 8115,
        "marker": "x",
        "color": "green",
        "alpha": 1.0,
        "Parameters": "135M",
        "Cost": None
    },
    {
        "exp_name": "SmolLM2-135M-Instruct",
        "source": None,
        "f1_score_ground_truth": 0.07527801539777589,
        "f1_score_generation": 0.4447391688770999,
        "latency": 35354,
        "marker": "x",
        "color": "orange",
        "alpha": 1.0,
        "Parameters": "135M",
        "Cost": None
    },
]


def plot_easy_results():
    # Plot ground truth accuracy
    fig = plt.figure(figsize=(15, 8))  # Increased figure size
    gs = plt.GridSpec(1, 2, width_ratios=[1.5, 1])  # Adjusted ratio for more legend space
    ax = fig.add_subplot(gs[0])
    
    # Separate data points by training source
    external_api_points = []
    local_llama_points = []
    no_finetuning_points = []
    
    for res_item in results_easy:
        point = ax.scatter(
            res_item["latency"], 
            res_item["acc_ground_truth"], 
            marker=res_item["marker"], 
            color=res_item["color"], 
            alpha=res_item["alpha"], 
            s=100, 
            label=res_item["exp_name"]
        )
    

        if res_item["source"] is None:
            no_finetuning_points.append(point)
        elif res_item["source"] == "external API":
            external_api_points.append(point)
        elif res_item["source"] == "local Llama 3.3 70B":
            local_llama_points.append(point)
    
    ax.set_xscale('log')
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Ground Truth')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Create legend space
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')  # Hide the axes
    
    # Create three legend sections
    first_legend = legend_ax.legend(external_api_points, 
                                  [p.get_label() for p in external_api_points],
                                  title="GPT-4o-mini teacher (External API)",
                                  loc='upper left')
    legend_ax.add_artist(first_legend)
    
    second_legend = legend_ax.legend(local_llama_points,
                                   [p.get_label() for p in local_llama_points],
                                   title="Llama-3.3-70B teacher (local)",
                                   loc='center left')
    legend_ax.add_artist(second_legend)
    
    legend_ax.legend(no_finetuning_points,
                    [p.get_label() for p in no_finetuning_points],
                    title="No Finetuning",
                    loc='lower left')
    
    plt.tight_layout(pad=3.0)  # Increased padding
    out_dir = os.path.join(os.path.dirname(__file__), "experiments/easy")
    plt.savefig(os.path.join(out_dir, "acc_vs_ground_truth.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_combined_experiments():
    # Create figure with three subplots
    fig = plt.figure(figsize=(25, 8))  # Wider figure to accommodate three sections
    gs = plt.GridSpec(1, 3, width_ratios=[1, 0.4, 1])  # Middle section for legend
    
    plt.rcParams.update({
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.title_fontsize': 21,
        'legend.fontsize': 20,
    })

    # Create subplots with more padding
    ax1 = fig.add_subplot(gs[0])  # Left plot for Experiment 1
    ax2 = fig.add_subplot(gs[2])  # Right plot for Experiment 2
    legend_ax = fig.add_subplot(gs[1])  # Middle section for legend
    legend_ax.axis('off')  # Hide the axes for legend section
    
    # Separate data points by training source
    external_api_points = []
    local_llama_points = []
    no_finetuning_points = []
    
    # Plot Experiment 1 (easy results)
    for res_item in results_easy:
        point = ax1.scatter(
            res_item["latency"], 
            res_item["acc_ground_truth"], 
            marker=res_item["marker"], 
            color=res_item["color"], 
            alpha=res_item["alpha"], 
            s=100, 
            label=res_item["exp_name"]
        )
        
        if res_item["source"] is None:
            no_finetuning_points.append(point)
        elif res_item["source"] == "external API":
            external_api_points.append(point)
        elif res_item["source"] == "local Llama 3.3 70B":
            local_llama_points.append(point)
    
    # Plot Experiment 2 (multi-tags results)
    for res_item in results_multi_tags:
        point = ax2.scatter(
            res_item["latency"], 
            res_item["f1_score_ground_truth"], 
            marker=res_item["marker"], 
            color=res_item["color"], 
            alpha=res_item["alpha"], 
            s=100, 
            label=res_item["exp_name"]
        )
    
    # Configure both plots
    for ax, title, ylabel in [(ax1, "Experiment 1: Restaurant Classification", "Accuracy"), 
                             (ax2, "Experiment 2: Tag Labelling", "F1 Score")]:
        ax.set_xscale('log')
        ax.set_xlabel('Latency (seconds)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", ls="-", alpha=0.2)

    # put the axis on the right side for ax2
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    
    # Create single legend with three sections
    first_legend = legend_ax.legend(external_api_points, 
                                  [p.get_label() for p in external_api_points],
                                  title="GPT-4o-mini teacher (External API)",
                                  loc='upper center')
    legend_ax.add_artist(first_legend)
    
    second_legend = legend_ax.legend(local_llama_points,
                                   [p.get_label() for p in local_llama_points],
                                   title="Llama-3.3-70B teacher (local)",
                                   loc='center')
    legend_ax.add_artist(second_legend)
    
    legend_ax.legend(no_finetuning_points,
                    [p.get_label() for p in no_finetuning_points],
                    title="No Finetuning",
                    loc='lower center')
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots
    
    out_dir = os.path.join(os.path.dirname(__file__), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "combined_experiments.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_taken_components_vs_pretrained_multitag():
    steps = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

    # Self-trained components (constants)
    collect_samples = 8115.68548116
    generate_embeddings = 555
    training_time = 18.250106811523438
    time_per_inference_self_trained = 67.38594794273376 * (1 / (124762 - 1024))

    # Pretrained inference time per step
    time_per_inference_pretrained = 29955 / 124762

    # Prepare data for plotting
    self_collect = []
    self_embed = []
    self_train = []
    self_infer = []
    pre_infer = []

    for i in steps:
        self_collect.append(collect_samples)
        self_embed.append(generate_embeddings)
        self_train.append(training_time)
        self_infer.append(time_per_inference_self_trained * i)
        pre_infer.append(time_per_inference_pretrained * i)

    # Bar positions
    x = np.arange(len(steps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    # Self-trained stacked bars (ours)
    p1 = ax.bar(x - width/2, self_collect, width, label='Generate Training Samples', color='tab:blue')
    p2 = ax.bar(x - width/2, self_embed, width, bottom=self_collect, label='Generate Embeddings', color='tab:orange')
    p3 = ax.bar(x - width/2, self_train, width, bottom=np.array(self_collect)+np.array(self_embed), label='Training', color='tab:green')
    p4 = ax.bar(x - width/2, self_infer, width, bottom=np.array(self_collect)+np.array(self_embed)+np.array(self_train), label='Inference', color='tab:red')

    # Pretrained bar (only inference)
    p5 = ax.bar(x + width/2, pre_infer, width, label='Inference', color='tab:purple', edgecolor='black', hatch='//')

    # Custom legends
    from matplotlib.patches import Patch
    ours_legend = [Patch(facecolor='tab:blue', edgecolor='black', label='Generate Training Samples (Llama-3.3-70B)'),
                   Patch(facecolor='tab:orange', edgecolor='black', label='Generate Embeddings'),
                   Patch(facecolor='tab:green', edgecolor='black', label='Training'),
                   Patch(facecolor='tab:red', edgecolor='black', label='Inference')]
    pretrained_legend = [Patch(facecolor='tab:purple', edgecolor='black', hatch='//', label='Inference (Pretrained)')]

    ax.set_xlabel('Number Rows')
    ax.set_ylabel('Time (s)')
    ax.set_title('Latency vs Rows of Data Processed')
    ax.set_xticks(x)
    ax.set_xticklabels(steps)

    # Place two legends
    leg1 = ax.legend(handles=ours_legend, title='ScaleLLM', loc='upper left')
    leg2 = ax.legend(handles=pretrained_legend, title='Pretrained', loc='upper right')
    ax.add_artist(leg1)  # Add the first legend back

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "experiments/multi_tags")
    plt.savefig(os.path.join(out_dir, "time_taken_components_vs_steps.png"))
    

    


def compare_hard_experiments():
    # Load experiment info
    json_path = os.path.join(os.path.dirname(__file__), "experiments/multi_tags/multi_tags_experiment_info_lookup.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Helper: get all experiments with given max_iters and max_train_samples
    def filter_exps(max_iters=None, max_train_samples=None):
        result = []
        for exp in data.values():
            if (max_iters is None or exp["max_iters"] == max_iters) and \
               (max_train_samples is None or exp["max_train_samples"] == max_train_samples):
                result.append(exp)
        return result

    # 1. Fixed max iters (500), varying max_train_samples
    exps_500iters = filter_exps(max_iters=500)
    # Separate baseline (max_train_samples=None)
    baseline_500 = [e for e in exps_500iters if e["max_train_samples"] is None]
    assert len(baseline_500) == 1
    exps_500iters = [e for e in exps_500iters if e["max_train_samples"] is not None]
    exps_500iters = sorted(exps_500iters, key=lambda x: x["max_train_samples"])

    train_samples = [e["max_train_samples"] for e in exps_500iters]
    precision = [e["val_precision"] for e in exps_500iters]
    recall = [e["val_recall"] for e in exps_500iters]
    f1 = [e["val_f1"] for e in exps_500iters]
    training_time_500iters = [e["training_time"] for e in exps_500iters]

    # Baseline values (if present)
    if baseline_500:
        baseline = baseline_500[0]
        baseline_label = "Baseline (all data)"
        baseline_train_samples = [max(train_samples) * 2]  # Plot to the right of the last point
        baseline_precision = [baseline["val_precision"]]
        baseline_recall = [baseline["val_recall"]]
        baseline_f1 = [baseline["val_f1"]]
        baseline_training_time = [baseline["training_time"]]
    else:
        baseline_label = None

    # Plot: Metrics vs max_train_samples (log x)
    plt.figure(figsize=(8,6))
    plt.plot(train_samples, precision, marker='o', label='Precision')
    plt.plot(train_samples, recall, marker='o', label='Recall')
    plt.plot(train_samples, f1, marker='o', label='F1')
    if baseline_500:
        plt.axvline(baseline_train_samples[0], color='k', linestyle='--', alpha=0.5)
        plt.plot(baseline_train_samples, baseline_precision, marker='s', color='C0', label=f'Precision {baseline_label}')
        plt.plot(baseline_train_samples, baseline_recall, marker='s', color='C1', label=f'Recall {baseline_label}')
        plt.plot(baseline_train_samples, baseline_f1, marker='s', color='C2', label=f'F1 {baseline_label}')
    plt.xlabel("max_train_samples")
    plt.ylabel("Score")
    plt.title("Metrics vs max_train_samples (max_iters=500)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xscale('log')
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "experiments/multi_tags")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "metrics_vs_max_train_samples_max_iters_500.png"))
    plt.show()

    # Plot: Metrics vs training_time for fixed max_iters=500, varying max_train_samples
    plt.figure(figsize=(8,6))
    plt.plot(training_time_500iters, precision, marker='o', label='Precision')
    plt.plot(training_time_500iters, recall, marker='o', label='Recall')
    plt.plot(training_time_500iters, f1, marker='o', label='F1')
    if baseline_500:
        plt.axvline(baseline_training_time[0], color='k', linestyle='--', alpha=0.5)
        plt.plot(baseline_training_time, baseline_precision, marker='s', color='C0', label=f'Precision {baseline_label}')
        plt.plot(baseline_training_time, baseline_recall, marker='s', color='C1', label=f'Recall {baseline_label}')
        plt.plot(baseline_training_time, baseline_f1, marker='s', color='C2', label=f'F1 {baseline_label}')
    plt.xlabel("Training Time (s)")
    plt.ylabel("Score")
    plt.title("Metrics vs Training Time (max_iters=500, varying max_train_samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_vs_training_time_max_iters_500.png"))
    plt.show()

    # 2. Fixed max_train_samples (8192), varying max_iters
    exps_8192samples = filter_exps(max_train_samples=8192)
    exps_8192samples = sorted(exps_8192samples, key=lambda x: x["max_iters"])
    max_iters = [e["max_iters"] for e in exps_8192samples]
    precision = [e["val_precision"] for e in exps_8192samples]
    recall = [e["val_recall"] for e in exps_8192samples]
    f1 = [e["val_f1"] for e in exps_8192samples]
    training_time = [e["training_time"] for e in exps_8192samples]

    # Metrics vs max_iters
    plt.figure(figsize=(8,6))
    plt.plot(max_iters, precision, marker='o', label='Precision')
    plt.plot(max_iters, recall, marker='o', label='Recall')
    plt.plot(max_iters, f1, marker='o', label='F1')
    plt.xlabel("max_iters")
    plt.ylabel("Score")
    plt.title("Metrics vs max_iters (max_train_samples=8192)")
    plt.legend()
    plt.grid(True)
    plt.xticks(max_iters, [str(x) for x in max_iters])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_vs_max_iters_max_train_samples_8192.png"))
    plt.show()

    # Metrics vs training_time
    plt.figure(figsize=(8,6))
    plt.plot(training_time, precision, marker='o', label='Precision')
    plt.plot(training_time, recall, marker='o', label='Recall')
    plt.plot(training_time, f1, marker='o', label='F1')
    plt.xlabel("Training Time (s)")
    plt.ylabel("Score")
    plt.title("Metrics vs Training Time (max_train_samples=8192)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_vs_training_time_max_train_samples_8192.png"))
    plt.show()


if __name__ == "__main__":
    # compare_hard_experiments()
    # plot_easy_results() 
    plot_time_taken_components_vs_pretrained_multitag()
    # plot_combined_experiments()