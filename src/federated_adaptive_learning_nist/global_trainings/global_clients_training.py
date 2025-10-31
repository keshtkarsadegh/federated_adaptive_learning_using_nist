import json
from pathlib import Path

import torch
from torch import nn
from torch import optim
from src.federated_adaptive_learning_nist.outliers.global_selected_outliers_training import global_outliers_training
from src.federated_adaptive_learning_nist.data_utils import NistDataset,NistPath
from src.federated_adaptive_learning_nist.model import FlexibleCNN
base_learning_rate = 1e-3
base_weight_decay = 1e-4


"""
Global client training for federated adaptive learning on NIST.

This script orchestrates training of a global model that incorporates both 
global writers and selected outlier clients. It sets up model, optimizer, 
paths, and dataset splits, and delegates the actual training loop to 
`global_outliers_training`.

Main components:
    - GlobalClientsTraining: class wrapping initialization and training
    - Uses NistDataset for data loading and FlexibleCNN as base model
    - Stores results, model checkpoints, and writer splits under NIST_EXPERIMENTS
"""
project_results_dir=Path(__file__).resolve().parents[3]/"results"


class GlobalClientsTraining:
    """
    Wrapper for training a global model on NIST data, combining global writers
    with selected outlier clients.

    Responsibilities:
        - Initialize model, optimizer, and loss function
        - Manage experiment paths and JSON writer splits
        - Load selected outliers and global writer IDs
        - Run training using global_outliers_training

    Attributes:
        selected_outliers (list): IDs of selected outlier clients.
        global_writers (list): IDs of global writers.
        local_writers (list): IDs of local writers.
        model (torch.nn.Module): The global CNN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        results_path (Path): Directory for experiment results.
        model_path (Path): Directory for saving model checkpoints.
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
        """
        Initialize result directories for the project.

        Creates the following if they do not exist:
            - results/global_clients_results/
            - results/global_clients_model/
            - results/outliers/
            - results/writer_split.json (expected file)
        """

        self.results_path.mkdir(parents=True,exist_ok=True)
        (project_results_dir).mkdir(parents=True, exist_ok=True)
        (project_results_dir/"outliers").mkdir(parents=True, exist_ok=True)


    def get_writers_split(self):
        """
        Load writer split information from JSON.

        Reads writer_split.json and assigns:
            - self.local_writers
            - self.global_writers
        """

        with open(self.writer_split_path, "r") as f:
            writers_split = json.load(f)
        self.local_writers=writers_split["local_writers"]
        self.global_writers=writers_split["global_writers"]
    def get_selected_outliers(self):
        """
        Load IDs of selected outlier clients.

        Reads selected_outliers.json and assigns the list to self.selected_outliers.
        """

        with open(self.selected_outliers_path, "r") as f:
            self.selected_outliers = json.load(f)


    def train(self, epochs):
        """
        Train global model with global writers and selected outliers.

        Builds datasets for:
            - Joint training set of global writers + outliers
            - Validation set
            - Global writers test set
            - Outliers test set

        Delegates the training loop to `global_outliers_training`.

        Args:
            epochs (int): Number of training epochs.
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

