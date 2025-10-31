import copy
import json
from tqdm import tqdm
import torch

from torch import optim, nn
import matplotlib.pyplot as plt

from src.federated_adaptive_learning_nist.model import FlexibleCNN
from src.federated_adaptive_learning_nist.nist_logger import NistLogger
from src.federated_adaptive_learning_nist.data_utils import NistDataset

# === Paths ===

base_learning_rate = 1e-3
base_weight_decay = 1e-4
"""
outlier_training.py

Purpose:
    Train personalized models for a set of outlier writers in the NIST dataset.
    Each outlier is trained independently, evaluated on global and client-level
    test sets, and the results are aggregated.

Key Features:
    - Builds per-writer CNN models (FlexibleCNN) with independent training loops.
    - Tracks and logs training/validation accuracy across epochs.
    - Selects the best checkpoint per writer based on validation accuracy.
    - Evaluates best models on:
        * Global writer test set
        * All selected outlier clients' test set
    - Saves results to JSON and plots accuracy curves for each writer.

Inputs:
    - global_writers: list of IDs for global writers used as test reference.
    - selected_5_writers: list of outlier writer IDs for training.
    - batch_size: mini-batch size for loaders.
    - num_epochs: number of training epochs.
    - outliers_results_path: directory to save JSON results and plots.

Outputs:
    - outlier_training_results.json (per-writer metrics)
    - <writer_id>_accuracy_plot.png (training vs validation + test refs)

Dependencies:
    - torch, tqdm, matplotlib
    - src.federated_adaptive_learning_nist.model.FlexibleCNN
    - src.federated_adaptive_learning_nist.data_utils.NistDataset
    - src.federated_adaptive_learning_nist.nist_logger.NistLogger
"""


def outliers_train(global_writers,selected_5_writers,batch_size,num_epochs,outliers_results_path):
    """
        Train and evaluate CNN models for a group of selected outlier writers.

        Args:
            global_writers (list[str] or list[int]): Writer IDs considered "global",
                used to build the global test loader for evaluation.
            selected_5_writers (list[str] or list[int]): IDs of outlier writers
                to train personalized models for.
            batch_size (int): Mini-batch size for training and evaluation.
            num_epochs (int): Number of epochs to train each outlier model.
            outliers_results_path (Path): Directory where plots and JSON
                metrics will be saved.

        Workflow:
            1. For each writer in `selected_5_writers`:
                - Build a new FlexibleCNN.
                - Train with writer-specific data (60/40 train/val split).
                - Track per-epoch training/validation accuracy.
                - Keep the checkpoint with the best validation accuracy.
            2. Evaluate the best checkpoint on:
                - Global test set (`global_writers`)
                - All selected outlier test set (`selected_5_writers`)
            3. Log results and plot accuracy curves.

        Side Effects:
            - Saves PNG plots per writer: "<writer_id>_accuracy_plot.png"
            - Aggregates all metrics into JSON: "outlier_training_results.json"
            - Logs training/validation/test results via NistLogger.

        Returns:
            None. All results are persisted as files and logs.
        """
    #`` === Track results ===
    results = {}
    nist_dataset=NistDataset()

    # === Train per writer ===
    for writer_id in selected_5_writers:
        best_client_model=None
        best_val_acc=-1
        client_model=FlexibleCNN(num_classes=10, complexity=3, dropout_rate=0.1, use_dropout=True,regularization=False)
        client_model = client_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = optim.Adam(client_model.parameters(), lr=base_learning_rate, weight_decay=base_weight_decay)
        criterion =  nn.CrossEntropyLoss()

        # === Load personal data ===
        train_loader, eval_loader,_ = nist_dataset.build_dataset(writer_id,train_rate=0.6,eval_rate=0.4,batch_size=batch_size)


        train_accuracies = []
        val_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):

            client_model.train()
            correct, total = 0, 0

            for images, labels in train_loader:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                optimizer.zero_grad()
                outputs = client_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_accuracies.append(train_acc)

            client_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in eval_loader:
                    images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = client_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            val_accuracies.append(val_acc)

            NistLogger.info(f"[{writer_id}] Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_client_model=copy.deepcopy(client_model)
        client_model = copy.deepcopy(best_client_model)
        # (Optional) Move to device
        client_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        client_model.eval()  # If you
        # === Evaluate on full global writer test set ===
        _, _,global_test_loader = nist_dataset.build_dataset(global_writers,train_rate=0.0,eval_rate=0.0,batch_size=batch_size)


        global_test_acc = 0.0
        correct, total = 0, 0
        if global_test_loader and len(global_test_loader.dataset) > 0:
            with torch.no_grad():
                for images, labels in global_test_loader:
                    images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = client_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            if total > 0:
                global_test_acc = correct / total
        _, _,all_client_test_loader = nist_dataset.build_dataset(selected_5_writers,train_rate=0.0,eval_rate=0.0,batch_size=batch_size)

        client_test_acc = 0.0
        correct, total = 0, 0
        if all_client_test_loader and len(all_client_test_loader.dataset) > 0:
            with torch.no_grad():
                for images, labels in all_client_test_loader:
                    images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = client_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            if total > 0:
                client_test_acc = correct / total
        results[writer_id] = {
            "train_accuracies": train_accuracies,
            "eval_accuracies": val_accuracies,
            "all_client_test_accuracies": client_test_acc,
            "global_test_accuracy": global_test_acc
        }

        # === Plot accuracy ===
        plt.figure()
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.axhline(y=client_test_acc, color='b', linestyle='-', label=f"Test Accuracy on All Clients Data : {client_test_acc:.2f}")
        plt.axhline(y=global_test_acc, color='r', linestyle='--', label=f"Test Accuracy on Global Data: {global_test_acc:.2f}")
        plt.title(f"Accuracy – {writer_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outliers_results_path / f"{writer_id}_accuracy_plot.png")
        plt.close()

    # === Save accuracy data as JSON ===
    json_path =outliers_results_path / "outlier_training_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    NistLogger.info(f"\nSaved all accuracy data to: {json_path}")
