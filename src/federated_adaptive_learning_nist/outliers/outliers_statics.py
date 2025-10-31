
import json
import random

import matplotlib.pyplot as plt

from src.federated_adaptive_learning_nist.nist_logger import NistLogger

"""
outlier_selection.py

Purpose:
    Identify and visualize low-performing (low-accuracy) clients/writers
    from global evaluation results in federated adaptive learning.

Key Features:
    - Loads per-client accuracy metrics on a global test set.
    - Ranks clients by accuracy (descending).
    - Selects the bottom 0.5% of writers (at least 1).
    - Randomly samples 5 writers from this bottom group (seeded).
    - Saves selected outlier IDs as JSON.
    - Produces plots:
        * All writers with bottom 0.5% highlighted.
        * Bottom 0.5% subset with selected writers emphasized.

Inputs:
    - clients_acc_on_global_path: JSON file with client accuracies.
    - selected_outliers_path: Output file for selected outlier IDs.
    - outliers_result_path: Directory for saving plots.

Outputs:
    - selected_outliers.json (chosen writer IDs)
    - all_none_contributor_writers.png
    - bottom_005_selected_writers.png

Dependencies:
    - matplotlib, json, random
    - src.federated_adaptive_learning_nist.nist_logger.NistLogger
"""


def get_low_acc_writers(clients_acc_on_global_path,selected_outliers_path,outliers_result_path,seed=42):
    """
        Select and visualize low-performing clients/writers.

        Args:
            clients_acc_on_global_path (Path or str):
                Path to JSON file containing client accuracies, formatted as
                a list of dicts: [{client_id: acc}, ...].
            selected_outliers_path (Path or str):
                Path to JSON file where selected outlier IDs will be saved.
            outliers_result_path (Path):
                Directory where plots will be saved.
            seed (int, optional):
                Random seed for reproducibility when sampling outliers.
                Defaults to 42.

        Process:
            1. Load accuracies and flatten to (client_id, acc) tuples.
            2. Sort clients by accuracy (descending).
            3. Take bottom 0.5% of writers (at least 1).
            4. Randomly select 5 writers from this group (or fewer if not enough).
            5. Save selected writer IDs to JSON.
            6. Generate two plots:
                - All writers with bottom 0.5% highlighted.
                - Bottom 0.5% only with chosen 5 annotated.

        Returns:
            list[str]:
                List of selected writer IDs (outliers).
        """
    # === Load and sort writers ===
    with open(clients_acc_on_global_path, "r") as f:
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
    seed = int(str(seed))  # ✅ Ensure it's int-compatible
    random.seed(seed)  # You can choose any integer seed value
    selected_5_writers = random.sample(bottom_005_percent_writers, n_sample)
    selected_outliers = [selected_5_writers[i][0] for i in range(n_sample)]

    with open(selected_outliers_path, "w") as f:
        json.dump(selected_outliers, f, indent=2)
    NistLogger.info(f"Low Acc Writers: {len(selected_outliers)}, saved at {selected_outliers_path}")

    # ---- Plot 1: All clients, highlight bottom 0.2% ----
    all_accs = [acc for _, acc in sorted_writers]
    highlight_005_accs = [acc for _, acc in bottom_005_percent_writers]
    highlight_005_indices = [i for i, (_, acc) in enumerate(sorted_writers) if (_, acc) in bottom_005_percent_writers]

    plt.figure(figsize=(12, 6))
    plt.plot(all_accs, label="All None Contributor Writers", color='gray')
    plt.scatter(highlight_005_indices, highlight_005_accs, color='red', label='Bottom 0.05%', zorder=5)
    plt.title("Writer Accuracies with Bottom 0.05% Highlighted")
    plt.xlabel("Writer Index (sorted by accuracy)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outliers_result_path/"all_none_contributor_writers.png")

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
            x, y,  # position
            f"{y:.2f}",  # text: show accuracy (2 decimals)
            fontsize=16,  # font size for labels
            ha='center',  # horizontal alignment
            va='bottom'  # vertical alignment
        )
    plt.title("Bottom 0.05% Writers with Selected Writers Highlighted")
    plt.xlabel("Writer Index (within bottom 0.05%)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outliers_result_path/"bottom_005_selected_writers.png")
    return selected_outliers


