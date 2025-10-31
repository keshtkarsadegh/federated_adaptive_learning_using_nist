import json
import numpy as np
from pathlib import Path


def summarize_outlier_training_results(input_path: Path, output_path: Path | None = None) -> dict:
    """
    Summarize outlier training results:
      Per client:
        - best_eval_accuracy (max of eval_accuracies)
        - global_test_accuracy
        - all_client_test_accuracies
      Overall means:
        - mean_global_test_accuracy
        - mean_all_client_test_accuracies
        - mean_best_eval_accuracy
    """

    # --- Load ---
    with open(input_path, "r") as f:
        results = json.load(f)

    summary = {}

    # --- Per-client metrics ---
    for client_id, data in results.items():
        eval_accuracies = data.get("eval_accuracies", [])
        best_eval_acc = max(eval_accuracies) if eval_accuracies else None

        summary[client_id] = {
            "global_test_accuracy": data["global_test_accuracy"],
            "all_client_test_accuracies": data["all_client_test_accuracies"],
            "best_eval_accuracy": best_eval_acc,
        }

    # --- Overall means ---
    mean_global_test_acc = np.mean([v["global_test_accuracy"] for v in summary.values()])
    mean_all_clients_test_acc = np.mean([v["all_client_test_accuracies"] for v in summary.values()])
    mean_best_eval_acc = np.mean([v["best_eval_accuracy"] for v in summary.values()])

    summary["__overall_means__"] = {
        "mean_global_test_accuracy": mean_global_test_acc,
        "mean_all_client_test_accuracies": mean_all_clients_test_acc,
        "mean_best_eval_accuracy": mean_best_eval_acc,
    }

    # --- Save ---
    if output_path is None:
        output_path = input_path.parent / "outlier_summary.json"

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to: {output_path}")
    return summary

from pathlib import Path

summary = summarize_outlier_training_results(
    Path("outlier_training_results.json")
)

print(json.dumps(summary["__overall_means__"], indent=2))
