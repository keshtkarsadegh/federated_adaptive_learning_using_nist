import json
import random

from matplotlib import pyplot as plt

with open("clients_acc_on_global.json", "r") as f:
    writers_acc_on_global = json.load(f)

    # Convert to flat list of tuples: [(client_id, acc), ...]
flattened = [(k, v) for entry in writers_acc_on_global for k, v in entry.items()]

# Sort by accuracy descending
sorted_writers = sorted(flattened, key=lambda x: x[1], reverse=True)
# Bottom 2% count
n_total = len(sorted_writers)
n_bottom = max(1, int(0.005 * n_total))  # at least 1

# Get bottom 2% slice
bottom_005_percent_writers = sorted_writers[-n_bottom:]

# Randomly sample 5 (or fewer if not enough)
n_sample = min(5, len(bottom_005_percent_writers))
random.seed(42)  # You can choose any integer seed value
selected_5_writers = random.sample(bottom_005_percent_writers, n_sample)
selected_outliers = [selected_5_writers[i][0] for i in range(n_sample)]


# ---- Plot 1: All clients, highlight bottom 0.2% ----
all_accs = [acc for _, acc in sorted_writers]
highlight_005_accs = [acc for _, acc in bottom_005_percent_writers]
highlight_005_indices = [i for i, (_, acc) in enumerate(sorted_writers) if (_, acc) in bottom_005_percent_writers]



# ---- Plot 2: Bottom 0.2% only, highlight selected 5 ----
bottom_accs = [acc for _, acc in bottom_005_percent_writers]
selected_indices = [i for i, (_, acc) in enumerate(bottom_005_percent_writers) if (_, acc) in selected_5_writers]
selected_accs = [acc for i, (_, acc) in enumerate(bottom_005_percent_writers) if i in selected_indices]

# Set global font size
plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(12, 6))
plt.plot(bottom_accs, label="Bottom 0.05%", color='orange')
plt.scatter(selected_indices, selected_accs, color='blue', label='Selected Clients', zorder=5)

# Annotate each selected point
for i, (x, y) in enumerate(zip(selected_indices, selected_accs)):
    plt.text(
        x, y,                 # position
        f"{y:.2f}",           # text: show accuracy (2 decimals)
        fontsize=16,          # font size for labels
        ha='center',          # horizontal alignment
        va='bottom'           # vertical alignment
    )
plt.title("Bottom 0.05% Writers with Selected Writers Highlighted")
plt.xlabel("Writer Index (within bottom 0.05%)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bottom_005_selected_writers.png")
plt.show()