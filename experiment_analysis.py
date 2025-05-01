import json
import matplotlib.pyplot as plt
import numpy as np

# Load the experiment results
with open('experiments/easy_max_train_samples.json', 'r') as f:
    data = json.load(f)

# Extract data points
max_train_samples = []
best_val_accuracy = []

for exp_name, exp_data in data.items():
    samples = exp_data['max_train_samples']
    if samples is None:
        samples = 12000  # Approximate full dataset size
    max_train_samples.append(samples)
    best_val_accuracy.append(exp_data['best_val_accuracy'])

# Sort the data points by max_train_samples
sorted_indices = np.argsort(max_train_samples)
max_train_samples = np.array(max_train_samples)[sorted_indices]
best_val_accuracy = np.array(best_val_accuracy)[sorted_indices]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(max_train_samples, best_val_accuracy, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Training Samples')
plt.ylabel('Best Validation Accuracy')
plt.title('Validation Accuracy vs Training Set Size')
plt.grid(True, alpha=0.3)

# Add data point labels
for i, (x, y) in enumerate(zip(max_train_samples, best_val_accuracy)):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('experiments/validation_accuracy_vs_samples.png')
plt.close()
