import os
import json
import matplotlib.pyplot as plt
import numpy as np

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
    compare_hard_experiments()