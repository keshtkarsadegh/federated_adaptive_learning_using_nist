import copy
import json
from pathlib import Path

import torch


from src.federated_adaptive_learning_nist.model import  FlexibleCNN
from src.federated_adaptive_learning_nist.nist_logger import NistLogger
from src.federated_adaptive_learning_nist.data_utils import NistDataset,NistPath

project_results_dir=Path(__file__).resolve().parents[3]/"results"

"""
base_concurrent_runner.py

Purpose:
    Implements the concurrent federated training/evaluation loop
    shared by both:
        - standalone training runs
        - grid search sweeps (EWC, distillation, logit consistency,
          feature alignment, prox, and extreme cases).

Capabilities:
    - Load and evaluate a global baseline model.
    - Train selected outlier clients against the global model concurrently.
    - Aggregate client updates using supplied aggregation methods.
    - Track and store round-by-round accuracy metrics.
    - Support flexible scenarios via the `single_outlier` argument.

Key Usage Modes:
    1. Normal training runs:
        Train all selected outliers jointly with the global model.
    2. Grid search:
        Called by sweep scripts to benchmark multiple hyperparameter/aggregation setups.
    3. Extreme cases:
        Use `single_outlier` to restrict training to one or more specific clients
        (e.g., one-client, two-client, or duplicated-client scenarios).

Argument: `single_outlier`
    - "None" (string): default, include all selected outliers.
    - list[str]: restrict to specified IDs only.
    - Applies in BOTH training and grid search.
    - Supports testing stability and fairness in minimal-client setups.

Outputs:
    - Round-by-round accuracy histories.
    - Metrics JSON files under results/.
    - Accuracy plots and logs.
    - Updated global model after aggregation.
"""


class BaseConcurrentRunner:
    """
    Federated concurrent training runner for adaptive learning on NIST.

    Provides the shared loop for both grid search and normal training,
    handling:
        - loading models and dataset splits
        - selecting clients (outliers/global)
        - per-client training
        - federated aggregation
        - round-by-round accuracy tracking

    Attributes:
        trainer (object): Trainer instance (e.g., EWCTrainer, DistillationTrainer).
        device (str): 'cuda' if available, otherwise 'cpu'.
        nist_dataset (NistDataset): Dataset manager for clients.
        global_model (FlexibleCNN): Current global model.
        accuracies (list): History of (all_clients_acc, global_acc) per round.
        client_weights (list): State dicts from trained client models.
        clients_samples_counts (list): Sample counts per client.
        all_clients_test_loader (DataLoader): Evaluation loader for outliers.
        global_test_loader (DataLoader): Evaluation loader for global writers.

    Note:
        The `single_outlier` argument controls which clients participate:
            - "None": include all selected outliers.
            - list[str]: restrict to given IDs, useful for extreme cases.
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

        self.global_model = None
        self.accuracies = []
        self.selected_outliers = None
        self.client_weights = []
        self.clients_samples_counts = []
        self.all_clients_test_loader = None
        self.global_test_loader = None
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
        global_clients_results_path = project_results_dir/f"global_clients_results"
        global_results_path=project_results_dir/f"global_results"
        global_metrics_file=global_results_path/"global_metrics.json"


        global_clients_metric_file=global_clients_results_path / "global_clients_global_metrics.json"
        global_clients_all_metrics_file=global_clients_results_path /"global_clients_all_outliers_metrics.json"



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
           Load a pre-trained global FlexibleCNN from disk.

           Args:
               global_name (str): Name prefix for the saved model directory.

           Returns:
               FlexibleCNN: Model loaded onto the correct device.
           """
        path = project_results_dir / f"{global_name}_model"
        model = FlexibleCNN(num_classes=10, complexity=3, dropout_rate=0.1, use_dropout=True, regularization=False)
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        return model

    def get_writers_split(self):
        """
           Load the client split (local vs global writers) from writer_split.json.

           Returns:
               tuple[list[str], list[str]]: (local_writers, global_writers)
           """
        path = project_results_dir / "writer_split.json"
        with open(path, "r") as f:
            split = json.load(f)

        return split["local_writers"], split["global_writers"]

    def get_selected_outliers(self):
        """
            Load IDs of selected outlier clients from JSON.

            Returns:
                tuple[list[str], list]: (selected_outliers, empty placeholder list)
            """
        path = project_results_dir / "outliers" / "selected_outliers.json"
        with open(path, "r") as f:
            selected = json.load(f)
        return selected, []

    def get_test_loaders(self, selected_outliers, global_writers, batch_size):
        """
            Build evaluation DataLoaders for selected outliers and global writers.

            Args:
                selected_outliers (list[str]): Outlier IDs.
                global_writers (list[str]): Global writer IDs.
                batch_size (int): Batch size for evaluation.

            Returns:
                tuple[DataLoader, DataLoader]: (all_clients_loader, global_loader)
            """
        all_clients_loader = self.nist_dataset.build_dataset(selected_outliers, train_rate=0.0, eval_rate=0.0, batch_size=batch_size)[2]
        global_loader = self.nist_dataset.build_dataset(global_writers, train_rate=0.0, eval_rate=0.0, batch_size=batch_size)[2]
        return all_clients_loader, global_loader

    def aggregate(self, aggregate_method):
        """
            Aggregate client weights into the global model.

            Args:
                aggregate_method (callable): Function with signature
                    (global_weights, client_weights, clients_samples_counts)
                    -> aggregated_state_dict

            Updates:
                self.global_model with aggregated weights.
            """


        global_weights = {
            k: v for k, v in self.global_model.state_dict().items() if v.dtype.is_floating_point
        }

        client_weights = self.client_weights  # Already filtered when stored

        aggregated = aggregate_method(global_weights, client_weights, self.clients_samples_counts)

        self.global_model.load_state_dict(aggregated, strict=False)



    def run_round(self, batch_size, epochs):
        """
            Run a single round of federated training.

            Steps:
                - Evaluate current global model on global and outlier test sets.
                - Train each selected outlier client (copy of global model).
                - Collect trained weights and sample counts.
                - Append round accuracies to history.

            Args:
                batch_size (int): Training batch size.
                epochs (int): Epochs per client.

            Returns:
                bool: True if training should continue, False if stopped early.
            """
        self.trainer.set_model(self.global_model)
        aggr_global_acc = self.trainer.evaluate(self.global_test_loader)
        aggr_all_clients_acc = self.trainer.evaluate(self.all_clients_test_loader)

        self.accuracies.append((aggr_all_clients_acc, aggr_global_acc))

        self.client_weights.clear()
        self.clients_samples_counts.clear()
        # if aggr_global_acc < aggr_all_clients_acc or aggr_global_acc < 0.90:
        #     return False


        for writer in self.selected_outliers:
            trn, evl, _ = self.nist_dataset.build_dataset(writer, train_rate=0.6, eval_rate=0.4, batch_size=batch_size)
            sample_count = self.nist_dataset.get_sample_count(writer)
            self.clients_samples_counts.append(sample_count)

            client_model = copy.deepcopy(self.global_model)
            self.trainer.set_model(client_model)
            self.trainer.train(trn, evl, epochs)

            trained = self.trainer.get_model()  # ✅ trained is defined here
            float_state_dict = {k: v for k, v in trained.state_dict().items() if v.dtype.is_floating_point}
            self.client_weights.append(copy.deepcopy(float_state_dict))
        return True



# single_outlier=["f3503_07","f3503_07"]
    def simulate(self, exp_name, global_name, aggregate_method, batch_size=64, epochs=100, max_round=100,grid_Search=False):
        """
            Simulate federated concurrent training over multiple rounds.

            Args:
                exp_name (str): Experiment name (used for output directories).
                global_name (str): Identifier for loading global model weights.
                aggregate_method (callable): Function to aggregate client weights.
                batch_size (int, optional): Training batch size. Default=64.
                epochs (int, optional): Client epochs per round. Default=100.
                max_round (int, optional): Maximum federated rounds. Default=100.
                grid_Search (bool, optional): Whether called inside a grid search. Default=False.
                single_outlier (list[str] or str, optional):
                    - "None" → use all selected outliers (normal training/grid search).
                    - list[str] → restrict to given IDs (extreme cases).

            Returns:
                - If grid_Search=False:
                    tuple: (accuracy_history, global_clients_all_metrics_acc,
                            global_clients_metric_acc, Path to experiments)
                - If grid_Search=True:
                    tuple: (accuracy_history, project_results_dir)
            """
        results_path = project_results_dir / f"{exp_name}_results"
        self.selected_outliers, self.accuracies = self.get_selected_outliers()
        if self.single_outlier !="None":
            self.selected_outliers = [sub for sub in self.selected_outliers if sub in self.single_outlier]

        local_writers, global_writers = self.get_writers_split()
        self.all_clients_test_loader, self.global_test_loader = self.get_test_loaders(self.selected_outliers, global_writers, batch_size)
        self.global_model = self.load_global_model(global_name)
        for r in range(max_round):
            NistLogger.debug(f"Training round {r}")
            should_continue=self.run_round(batch_size, epochs)
            if not should_continue:
                break
            self.aggregate(aggregate_method)
        if not grid_Search:
            self.get_global_base_accuracy()
            return self.accuracies, self.global_clients_all_metrics_acc, self.global_clients_metric_acc, project_results_dir
        else:
            return self.accuracies, project_results_dir

