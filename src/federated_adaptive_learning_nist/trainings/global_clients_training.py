import json
from pathlib import Path

from torch import nn
import torch

from torch import optim

from src.federated_adaptive_learning_nist.outliers.global_selected_outliers_training import global_outliers_training

from src.federated_adaptive_learning_nist.data_utils import NistDataset, NistPath

from src.federated_adaptive_learning_nist.model import FlexibleCNN
base_learning_rate = 1e-3
base_weight_decay = 1e-4
project_results_dir=Path(__file__).resolve().parents[1]/"results"
"""
global_clients_training.py

Purpose:
    Provides a training routine for a combined model using both global writers
    and previously identified outlier clients. This is used as a baseline
    for federated adaptive learning with NIST.

Capabilities:
    - Loads global and local writer splits from disk.
    - Loads preselected outlier clients.
    - Builds datasets combining global + outliers.
    - Trains a global model with outliers included.
    - Evaluates and stores metrics/plots for global-only and outlier-only tests.

Outputs:
    - Trained global model weights under results/global_clients_model.
    - Metrics JSON files (global_clients_results/).
    - Accuracy plots for training, validation, and test performance.

Usage:
    $ python global_clients_training.py
    (runs training loop for global+outliers, saves metrics and plots)
"""


class GlobalClientsTraining:
    """
        Orchestrates training of a global model on a mix of outlier clients and
        global writers, producing a "global_clients" model and associated metrics.

        Attributes:
            selected_outliers (list[str]): Outlier client IDs loaded from JSON.
            global_writers (list[str]): Global writer IDs loaded from JSON.
            local_writers (list[str]): Local writer IDs loaded from JSON.
            model (FlexibleCNN): The CNN model to be trained.
            optimizer (torch.optim.Adam): Optimizer for training.
            criterion (nn.CrossEntropyLoss): Loss function.
            results_path (Path): Directory for saving results and plots.
            model_path (Path): Path for saving trained model weights.
            writer_split_path (Path): JSON path containing writer splits.
            selected_outliers_path (Path): JSON path containing selected outliers.

        Typical Workflow:
            1. Instantiate the class.
            2. Call `get_selected_outliers()` to load the outlier list.
            3. Call `train(epochs=...)` to train the global+outliers model.
        """
    def __init__(self,):
        self.selected_outliers = None
        self.global_writers = None
        self.local_writers = None
        self.metrics = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = FlexibleCNN(num_classes=10, complexity=3, dropout_rate=0.1, use_dropout=True,
                                     regularization=False)
        self.model.to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=base_learning_rate, weight_decay=base_weight_decay)
        self.learning_rate = base_learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.name="global_clients"
        self.data_path = NistPath()
        self.project_name = "federated_adaptive_learning_using_nist"
        self.results_path = project_results_dir/f"{self.name}_results"
        self.model_path = project_results_dir / f"{self.name}_model"
        self.writer_split_path = project_results_dir / "writer_split.json"
        self.selected_outliers_path=project_results_dir/"outliers"/"selected_outliers.json"

        self.path_init()
        self.get_writers_split()
    def path_init(self):
        self.results_path.mkdir(parents=True,exist_ok=True)
        (project_results_dir).mkdir(parents=True, exist_ok=True)
        (project_results_dir/"outliers").mkdir(parents=True, exist_ok=True)


    def get_writers_split(self):
        with open(self.writer_split_path, "r") as f:
            writers_split = json.load(f)
        self.local_writers=writers_split["local_writers"]
        self.global_writers=writers_split["global_writers"]
    def get_selected_outliers(self):
        with open(self.selected_outliers_path, "r") as f:
            self.selected_outliers = json.load(f)


    def train(self, epochs):
        """
            Train the global model on the combination of global writers + selected outliers.

            Args:
                epochs (int): Number of training epochs.

            Process:
                - Build dataset loaders for training (global+outliers), validation,
                  global-only test, and outlier-only test.
                - Pass data and model to global_outliers_training() for execution.
                - Saves trained model and metrics to disk.
            """
        nist_dataset = NistDataset()

        clients_outliers = self.selected_outliers + self.global_writers
        train_loader, eval_loader, _ = nist_dataset.build_dataset(clients_outliers, train_rate=0.6, eval_rate=0.4,
                                                                  batch_size=64)
        _, _, global_test_loader = nist_dataset.build_dataset(self.global_writers, train_rate=0.0, eval_rate=0.0,
                                                              batch_size=64)
        _, _, all_outliers_test_loader = nist_dataset.build_dataset(self.selected_outliers, train_rate=0.0,
                                                                    eval_rate=0.0, batch_size=64)
        global_outliers_training(self.model, train_loader, eval_loader, all_outliers_test_loader, global_test_loader,
                                 self.model_path, self.results_path,  num_epochs=epochs)


if __name__ == "__main__":
    gct=GlobalClientsTraining()
    gct.get_selected_outliers()
    gct.train( epochs=100)

