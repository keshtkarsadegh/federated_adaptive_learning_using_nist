import os
import json
import csv
import numpy as np

def process_json(filepath):
    """Load a JSON file and compute the required metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)

    client_accs = np.array([x[0] for x in data["accuracies"]])
    global_accs = np.array([x[1] for x in data["accuracies"]])

    min_global = global_accs.min()*100
    max_client = client_accs.max()*100

    # % of client accuracies above global_clients_all_metrics_acc
    percent_local_above_threshold = np.mean(client_accs > data["global_clients_all_metrics_acc"]) * 100

    # % of epochs where global accuracy > client accuracy
    global_drop = (global_accs[0] * 100- min_global)

    return {
        "scenario": data["scenario"],
        "metadata": data["metadata"],
        "agg_method_name": data["agg_method_name"],
        "min_acc_on_global_dataset": float(min_global),
        "max_acc_on_clients_dataset": float(max_client),
        "adaptation_stability": percent_local_above_threshold,
        "global_knowledge_drop": global_drop
    }

def collect_results_for_all_aggs(root_folder, output_csv="results.csv",saving=True):
    results = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.startswith("accuracies") and fname.endswith(".json"):
                filepath = os.path.join(dirpath, fname)
                try:
                    result = process_json(filepath)
                    results.append(result)
                except Exception as e:
                    print(f"Skipping {filepath}: {e}")
    if saving:
        if results:
            keys = results[0].keys()
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"✅ Results saved to {output_csv}")
        else:
            print("⚠️ No valid JSON files found.")
    else:
        return results

if __name__ == "__main__":
    root_folder = "."  # change this
    collect_results_for_all_aggs(root_folder, output_csv="aggregated_results.csv")
