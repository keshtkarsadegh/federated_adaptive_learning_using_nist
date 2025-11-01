import copy
import json
from pathlib import Path

import torch

from src.federated_adaptive_learning_nist.model import FlexibleCNN
from src.federated_adaptive_learning_nist.nist_logger import NistLogger
from src.federated_adaptive_learning_nist.data_utils import NistDataset,NistPath

project_results_dir=Path(__file__).resolve().parents[3]/"results"
"""
base_sequential_runner.py

Purpose:
    Implements the sequential federated training/evaluation loop
    shared by both:
        - standalone training runs
        - grid search sweeps (EWC, distillation, logit consistency,
          feature alignment, prox, and extreme cases).

Capabilities:
    - Load and evaluate a global baseline model.
    - Train selected outlier clients sequentially against the global model.
    - Aggregate client updates one-by-one using supplied aggregation methods.
    - Track and store round-by-round accuracy metrics.
    - Support flexible scenarios via the `single_outlier` argument.

Key Usage Modes:
    1. Normal training runs:
        Train all selected outliers in sequence with the global model.
    2. Grid search:
        Used by parameter sweep scripts to benchmark multiple setups.
    3. Extreme cases:
        Use `single_outlier` to restrict training to specific IDs
        (e.g., one-client, two-client, duplicated-client experiments).

Argument: `single_outlier`
    - "None" (string): include all selected outliers (default).
    - list[str]: restrict to specified IDs only.
    - Applies in BOTH training and grid search modes.

Outputs:
    - Round-by-round accuracy histories.
    - Metrics JSON files under results/.
    - Updated global model after aggregation.
"""

class BaseSequentialRunner:
    """
        Federated sequential training runner for adaptive learning on NIST.

        Provides the shared loop for both grid search and normal training,
        handling:
            - loading models and dataset splits
            - selecting clients (outliers/global)
            - sequential per-client training
            - aggregation after each client
            - round-by-round accuracy tracking

        Attributes:
            trainer (object): Trainer instance (e.g., EWCTrainer, DistillationTrainer).
            device (str): 'cuda' if available, otherwise 'cpu'.
            nist_dataset (NistDataset): Dataset manager for clients.
            global_model (FlexibleCNN): Current global model.
            accuracies (list): History of (all_clients_acc, global_acc) per round.
            client_weights (list): Single client’s weights (sequentially replaced).
            all_clients_test_loader (DataLoader): Evaluation loader for outliers.
            global_test_loader (DataLoader): Evaluation loader for global writers.
            final_local_all_clients_acc (float): Last-round outlier test accuracy.
            final_local_global_acc (float): Last-round global test accuracy.

        Note:
            The `single_outlier` argument controls which clients participate:
                - "None": include all selected outliers.
                - list[str]: restrict to given IDs (extreme cases).
            This applies equally to training runs and grid search experiments.
        """
    def __init__(self, trainer,single_outlier=None):
        self.single_outlier=single_outlier
        self.global_clients_all_metrics_acc = None
        self.global_clients_metric_acc = None
        self.global_metrics_acc = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(f"device: {self.device}")

        self.data_path = NistPath()
        self.nist_dataset = NistDataset()
        self.trainer = trainer
        self.global_model = FlexibleCNN(num_classes=10, complexity=3, dropout_rate=0.1, use_dropout=True, regularization=False)

        self.selected_outliers = None
        self.global_writers = None
        self.local_writers = None
        self.accuracies = []
        self.client_weights = []
        self.all_clients_test_loader = None
        self.global_test_loader = None

        self.final_local_all_clients_acc = None
        self.final_local_global_acc = None

    def get_global_base_accuracy(self):
        """
            Load baseline test accuracies from saved results.

            Reads JSON files under results/:
                - global_results/global_metrics.json
                - global_clients_results/global_clients_global_metrics.json
                - global_clients_results/global_clients_all_outliers_metrics.json

            Sets attributes:
                - self.global_metrics_acc
                - self.global_clients_metric_acc
                - self.global_clients_all_metrics_acc
            """
        global_clients_results_path = project_results_dir / f"global_clients_results"
        global_results_path = project_results_dir / f"global_results"
        global_metrics_file = global_results_path / "global_metrics.json"

        global_clients_metric_file = global_clients_results_path / "global_clients_global_metrics.json"
        global_clients_all_metrics_file = global_clients_results_path / "global_clients_all_outliers_metrics.json"

        def read_test_accuracy(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            return data.get("test_accuracy")

        if self.global_clients_all_metrics_acc is None or self.global_clients_metric_acc is None or self.global_metrics_acc is None:
            # Read test_accuracy from each file
            self.global_metrics_acc = read_test_accuracy(global_metrics_file)
            self.global_clients_metric_acc = read_test_accuracy(global_clients_metric_file)
            self.global_clients_all_metrics_acc = read_test_accuracy(global_clients_all_metrics_file)
    def load_global_model(self, global_name):
        """
           Load a pre-trained global FlexibleCNN from disk into self.global_model.

           Args:
               global_name (str): Name prefix for the saved model directory.
           """
        path = project_results_dir / f"{global_name}_model"
        self.global_model.load_state_dict(torch.load(path))
        self.global_model.to(self.device)

    def get_writers_split(self):
        """
           Load the client split (local vs global writers) from writer_split.json.

           Sets:
               - self.local_writers
               - self.global_writers
           """
        path = project_results_dir / "writer_split.json"
        with open(path, "r") as f:
            split = json.load(f)
        self.local_writers = split["local_writers"]
        self.global_writers = split["global_writers"]

    def get_selected_outliers(self,):
        """
            Load IDs of selected outlier clients from JSON.

            Args:
                single_outlier (list[str], optional):
                    If provided, restricts selection to those IDs only.

            Sets:
                - self.selected_outliers
            """
        path = project_results_dir/ "outliers" / "selected_outliers.json"
        with open(path, "r") as f:
            self.selected_outliers = json.load(f)
        if self.single_outlier:
            self.selected_outliers = [sub for sub in self.selected_outliers if sub in self.single_outlier]

    def get_test_loaders(self, batch_size):
        """
            Build evaluation DataLoaders for selected outliers and global writers.

            Args:
                batch_size (int): Batch size for evaluation.

            Sets:
                - self.all_clients_test_loader
                - self.global_test_loader
            """
        _, _, self.all_clients_test_loader = self.nist_dataset.build_dataset(self.selected_outliers, train_rate=0.0, eval_rate=0.0, batch_size=batch_size)
        _, _, self.global_test_loader = self.nist_dataset.build_dataset(self.global_writers, train_rate=0.0, eval_rate=0.0, batch_size=batch_size)




    def aggregate(self, aggregate_method, index=None):
        """
            Aggregate current client update into the global model.

            Args:
                aggregate_method (callable): Function with signature
                    (global_weights, client_weight, index, num_clients,
                     client_sample_count, all_clients_samples)
                    -> aggregated_state_dict
                index (int, optional): Position of client in sequential order.

            Updates:
                self.global_model with aggregated weights.
            """
        global_weights = {k: v.to(self.device) for k, v in self.global_model.state_dict().items()}

        assert isinstance(self.client_weights, list) and len(self.client_weights) == 1, \
            "Sequential aggregation expects exactly one client model at a time"

        client_weight = {k: v.to(self.device) for k, v in self.client_weights[0].items()}

        # === Extract optional metadata ===
        num_clients = len(self.selected_outliers)
        client_sample_count = self.nist_dataset.get_sample_count(self.selected_outliers[index])
        all_clients_samples = sum(self.nist_dataset.get_sample_count(cid) for cid in self.selected_outliers)

        # === Call aggregation function with proper kwargs ===
        aggregated = aggregate_method(
            global_weights=global_weights,
            client_weight=client_weight,
            index=index,
            num_clients=num_clients,
            client_sample_count=client_sample_count,
            all_clients_samples=all_clients_samples
        )

        self.global_model.load_state_dict(aggregated)

    def fl_sequence(self, writer_id, batch_size, epochs, aggregate_method, is_last,index):
        """
            Run sequential training for a single client.

            Steps:
                - Copy global model to client model.
                - Train client model for given epochs.
                - Collect client weights.
                - Aggregate into global model.

            Args:
                writer_id (str): Client ID.
                batch_size (int): Training batch size.
                epochs (int): Number of local training epochs.
                aggregate_method (callable): Function to combine client/global weights.
                is_last (bool): Whether this is the last client in sequence.
                index (int): Position index of client in selected_outliers.
            """
        client_model = copy.deepcopy(self.global_model)
        trn, evl, _ = self.nist_dataset.build_dataset(writer_id, train_rate=0.6, eval_rate=0.4, batch_size=batch_size)

        self.trainer.set_model(client_model)
        self.trainer.train(trn, evl, epochs)
        trained_model = self.trainer.get_model()
        self.client_weights = [trained_model.state_dict()]
        self.aggregate(aggregate_method,index=index)

    def fl_round(self, batch_size, epochs, aggregate_method):
        """
            Run one federated round over all selected outliers.

            Steps:
                - Evaluate global model on outlier and global test loaders.
                - Train each outlier sequentially using fl_sequence.
                - Aggregate after each client.

            Args:
                batch_size (int): Training batch size.
                epochs (int): Local training epochs.
                aggregate_method (callable): Aggregation method.

            Returns:
                bool: True if training should continue, False otherwise.
            """
        self.trainer.set_model(self.global_model)
        aggr_all_clients_acc = self.trainer.evaluate(self.all_clients_test_loader)
        aggr_global_acc = self.trainer.evaluate(self.global_test_loader)
        self.accuracies.append((aggr_all_clients_acc, aggr_global_acc))



        # if aggr_global_acc < aggr_all_clients_acc or aggr_global_acc < 0.90:
        #     return False

        for i, writer in enumerate(self.selected_outliers):
            is_last = (i == len(self.selected_outliers) - 1)
            self.fl_sequence(writer, batch_size, epochs, aggregate_method, is_last,index=i)

        return True
# single_outlier=["f3503_07","f3503_07"]
    def simulate(self, exp_name, global_name, aggregate_method, batch_size=64, epochs=100, max_round=20,grid_Search=False):
        """
            Simulate federated sequential training over multiple rounds.

            Args:
                exp_name (str): Experiment name (used for output directories).
                global_name (str): Identifier for loading global model weights.
                aggregate_method (callable): Function to aggregate weights.
                batch_size (int, optional): Training batch size. Default=64.
                epochs (int, optional): Client epochs per round. Default=100.
                max_round (int, optional): Maximum federated rounds. Default=20.
                grid_Search (bool, optional): Whether called inside a grid search. Default=False.
                single_outlier (list[str] or str, optional):
                    - "None" → use all selected outliers (normal runs).
                    - list[str] → restrict to given IDs (extreme cases).

            Returns:
                - If grid_Search=False:
                    tuple: (accuracy_history, global_clients_all_metrics_acc,
                            global_clients_metric_acc, project_results_dir)
                - If grid_Search=True:
                    tuple: (accuracy_history, project_results_dir)
            """
        results_path = project_results_dir / f"{exp_name}_results"
        self.get_writers_split()
        self.get_selected_outliers()
        self.load_global_model(global_name)
        self.get_test_loaders(batch_size)

        for r in range(max_round):
            NistLogger.debug(f"Training round {r}")
            should_continue = self.fl_round(batch_size, epochs, aggregate_method)
            if not should_continue:
                break
        if not grid_Search:
            self.get_global_base_accuracy()
            return self.accuracies, self.global_clients_all_metrics_acc,self.global_clients_metric_acc, project_results_dir

        else:
            return self.accuracies, project_results_dir

