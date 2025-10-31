import json
from pathlib import Path

import torch

from torch import nn
from tqdm import tqdm

from src.federated_adaptive_learning_nist.nist_logger import NistLogger
from src.federated_adaptive_learning_nist.data_utils import NistDataset

base_criterion = nn.CrossEntropyLoss()
"""
selecting_outliers.py

Purpose:
    Evaluate how well a trained global model performs on individual clients’ test sets,
    in order to identify low-performing ("outlier") clients in federated adaptive learning.

Key Features:
    - Iterates over a set of local writers (clients).
    - Evaluates global model accuracy on each client’s test set.
    - Collects per-client accuracy results into a list of dicts.
    - Supports caching results to avoid redundant evaluation.
    - Persists results to JSON for downstream selection/analysis.

Inputs:
    - global_model: Trained PyTorch model (e.g., CNN).
    - local_writers: List of client IDs.
    - clients_acc_on_global_path: Path to JSON file for saving/loading results.
    - force_generate: If True, recompute results even if JSON exists.

Outputs:
    - clients_acc_on_global.json (list of {client_id: accuracy})
    - Console logs per client and summary.

Dependencies:
    - torch, tqdm, json
    - src.federated_adaptive_learning_nist.nist_logger.NistLogger
    - src.federated_adaptive_learning_nist.data_utils.NistDataset
"""


def select_outliers(global_model,local_writers,clients_acc_on_global_path:Path,force_generate=False):
    """
        Evaluate a global model on all clients’ test sets and record their accuracies.

        Args:
            global_model (torch.nn.Module):
                Trained PyTorch model to be evaluated.
            local_writers (list[str]):
                List of client/writer IDs whose test sets will be used.
            clients_acc_on_global_path (Path):
                Path to JSON file for saving or loading evaluation results.
            force_generate (bool, optional):
                If True, recompute client accuracies even if the JSON file exists.
                If False (default), load cached results if available.

        Process:
            1. If cached JSON exists (and force_generate is False), load and return results.
            2. Otherwise:
                - For each client, build their test dataset.
                - Run evaluation with the global model (CrossEntropyLoss).
                - Compute accuracy per client.
                - Log and collect results into a list of dicts.
            3. Save results to JSON.

        Returns:
            list[dict]:
                A list of {client_id: accuracy} entries, one per client.
        """
    if not force_generate and  clients_acc_on_global_path.exists():
        NistLogger.info(f"Loading clients with accuracy on global model from {clients_acc_on_global_path}")
        with open(clients_acc_on_global_path, "r") as f:
            clients_acc_on_global = json.load(f)
        return clients_acc_on_global
# === Evaluate Only on Test Set Per Client ===
    clients_acc_on_global = []
    nist_dataset=NistDataset()

    for client_id in tqdm(local_writers, desc="Processing Clients"):
        _, _, test_loader = nist_dataset.build_dataset(client_id,train_rate=0.0,eval_rate=0.0,batch_size=64)

        if test_loader is None :
            NistLogger.info(f"Skipping client {client_id}: No test data.")
            continue

        global_model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = global_model(images)
                loss = base_criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total if test_total > 0 else 0.0

        NistLogger.info(f"Client '{client_id}' — Test Accuracy: {test_acc:.4f}")

        clients_acc_on_global.append({f"{client_id}": test_acc})

    NistLogger.info("\n📉 Lis of all clients with accuracy using trained global model")

    with open(clients_acc_on_global_path, "w") as f:
        json.dump(clients_acc_on_global, f)
    return clients_acc_on_global


