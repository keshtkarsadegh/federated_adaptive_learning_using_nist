import json
from pathlib import Path

import torch
from torch import nn
from matplotlib import pyplot as plt
from src.federated_adaptive_learning_nist.model import  FlexibleCNN
from src.federated_adaptive_learning_nist.nist_logger import NistLogger
from src.federated_adaptive_learning_nist.outliers.oulier_training import outliers_train
from src.federated_adaptive_learning_nist.outliers.outliers_statics import get_low_acc_writers
from src.federated_adaptive_learning_nist.outliers.selecting_outliers import select_outliers
from src.federated_adaptive_learning_nist.trainers.base_trainer import BaseTrainer
from src.federated_adaptive_learning_nist.global_trainings.fisher_util import compute_and_save_fisher_and_params
from src.federated_adaptive_learning_nist.data_utils import split_local_global_writers, NistDataset,NistPath

"""
Global training pipeline for federated adaptive learning on NIST.

This module coordinates:
    - Global model training and evaluation
    - Writer split generation (local vs global)
    - Outlier detection and training
    - Fisher information computation (for continual learning regularization)
    - Metrics logging and plotting

It serves as the entry point for preparing the global baseline and
supporting outlier-related experiments.
"""
project_results_dir=Path(__file__).resolve().parents[3]/"results"


class GlobalTraining:
    """
    Global training manager for NIST federated adaptive learning experiments.

    This class coordinates the baseline training of a global model on NIST
    writers, manages experiment paths, and integrates additional steps such as:
        - Generating local/global writer splits
        - Logging and plotting metrics
        - Selecting low-performing (outlier) clients
        - Training with global + outlier clients
        - Computing Fisher information for continual learning

    Attributes:
        model (FlexibleCNN): The global CNN model.
        trainer (BaseTrainer): Training utility for running experiments.
        local_writers (list): IDs of local writers.
        global_writers (list): IDs of global writers.
        selected_writers (list): IDs of selected outlier writers.
        results_path (Path): Directory for experiment results.
        model_path (Path): Directory for saving checkpoints.
        fisher_path (Path): Directory for Fisher information files.
    """

    def __init__(self,):
        self.metrics = None
        self.model = FlexibleCNN(num_classes=10, complexity=3, dropout_rate=0.1, use_dropout=True, regularization=False)
        self.local_writers = None
        self.global_writers = None
        self.selected_writers = None
        self.trainer = BaseTrainer()
        self.name="global"
        self.data_path = NistPath()
        self.project_name = "federated_adaptive_learning_using_nist"
        self.results_path = project_results_dir/f"{self.name}_results"
        self.model_path = project_results_dir/f"{self.name}_model"
        self.writer_split_path=project_results_dir/"writer_split.json"
        self.clients_acc_on_global_path=project_results_dir/"outliers"/"clients_acc_on_global.json"
        self.selected_outliers_path=project_results_dir/"outliers"/"selected_outliers.json"
        self.outliers_result_path=project_results_dir/"outliers"/"results"
        self.fisher_path=self.results_path/"fisher"
        self.path_init()




    def path_init(self):
        """
        Create required directories for results, models, and outliers.

        Ensures that the following exist:
            - results/global_results/
            - results/global_model/
            - results/outliers/
            - results/outliers/results/
        """

        (project_results_dir).mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        (project_results_dir/"outliers").mkdir(parents=True, exist_ok=True)
        self.outliers_result_path.mkdir(parents=True, exist_ok=True)

    def train(self,trn_loader,evl_loader,epochs,load_model=False):
        """
        Train or load the global model.

        If `load_model=True`, loads the model from disk. Otherwise trains from scratch
        using the provided train and validation loaders, then saves the model state.

        Args:
            trn_loader (DataLoader): Training dataset loader.
            evl_loader (DataLoader): Validation dataset loader.
            epochs (int): Number of epochs to train.
            load_model (bool): If True, load existing model instead of training.
        """

        if load_model:
            NistLogger.info(f"Loading model from {self.model_path}")
            self.model =self.trainer.load_model(self.model_path)
        else:

            self.trainer.set_model(self.model)
            self.trainer.train(train_loader=trn_loader, eval_loader=evl_loader, epochs=epochs)
            self.model=self.trainer.get_model()
            torch.save(self.model.state_dict(), self.model_path)
    def evaluate(self,tst_loader):
        """
        Evaluate the global model on a test set.

        Loads the saved state_dict and evaluates using BaseTrainer.

        Args:
            tst_loader (DataLoader): Test dataset loader.
        """

        self.model.load_state_dict(torch.load(self.model_path))
        self.trainer.set_model(self.model)
        self.trainer.evaluate(tst_loader)

    def generate_split_writers(self,global_size=0.03,seed=42):
        """
        Split writers into local and global sets and save to JSON.

        Uses `split_local_global_writers` to randomly partition dataset.

        Args:
            global_size (float): Proportion of writers to assign to the global set.
            seed (int): Random seed.

        Returns:
            tuple: (local_writers, global_writers)
        """

        local_writers, global_writers = split_local_global_writers(
            nist_data_path=str(self.data_path.by_write_data_path),
            global_size=global_size,seed=seed
        )
        with open(self.writer_split_path, "w") as f:
            json.dump({"local_writers": local_writers, "global_writers": global_writers}, f, indent=2)
        NistLogger.info(f"Local ({len(local_writers)}) and global ({len(global_writers)}) writers are stored.")
        self.global_writers = global_writers
        self.local_writers =local_writers
        return local_writers, global_writers
    def save_metrics_plots(self):
        """
        Save training/validation/test metrics and accuracy plots.

        - Saves metrics as JSON
        - Saves PNG plot of train/validation accuracy curves and test accuracy line
        """

        # === Save Metrics as JSON ===
        self.metrics = {
            "train_accuracies": self.trainer.train_accuracies,
            "val_accuracies": self.trainer.val_accuracies,
            "test_accuracy": self.trainer.test_acc
        }
        with open(self.results_path/f"{self.name}_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        NistLogger.info(f"Saved metrics JSON to {self.results_path/f"{self.name}_metrics.json"}")

        # === Save Accuracy Plot ===
        plt.figure()
        plt.plot(self.trainer.train_accuracies, label="Train Accuracy")
        plt.plot(self.trainer.val_accuracies, label="Validation Accuracy")
        plt.axhline(y=self.trainer.test_acc, color='r', linestyle='--', label=f"Test Accuracy on Global Data : {self.trainer.test_acc:.4f}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training / Validation Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_path/f"{self.name}_plots.png")
        NistLogger.info(f"Saved accuracy plot to {self.results_path/f"{self.name}_plots.png"}")
    def generate_selected_writers(self,seed=42):
        """
        Select underperforming (outlier) writers.

        Calls `get_low_acc_writers` using client accuracy statistics.

        Args:
            seed (int): Random seed.

        Returns:
            list: Selected outlier writer IDs.
        """

        self.selected_writers= get_low_acc_writers(self.clients_acc_on_global_path,self.selected_outliers_path,self.outliers_result_path,seed=seed)
        return self.selected_writers
    def get_outliers(self,force_generate=False):
        """
        Compute client accuracies to identify potential outliers.

        Args:
            force_generate (bool): If True, regenerate client accuracy stats even if file exists.
        """


        select_outliers(self.trainer.model, self.local_writers, self.clients_acc_on_global_path,force_generate=force_generate)

    def outliers_training(self,batch_size,epochs):
        """
        Train with global and selected outlier clients together.

        Delegates to `outliers_train`.

        Args:
            batch_size (int): Batch size.
            epochs (int): Number of epochs.
        """

        outliers_train(self.global_writers, self.selected_writers, batch_size=batch_size, num_epochs=epochs, outliers_results_path=self.outliers_result_path)

    def generate_fisher(self,trn_loader):
        """
        Compute Fisher information and save parameter snapshots.

        Runs `compute_and_save_fisher_and_params` using the provided training data.

        Args:
            trn_loader (DataLoader): Training dataset loader.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        compute_and_save_fisher_and_params(
    model=self.model,
    dataloader=trn_loader,  # use global data
    criterion=nn.CrossEntropyLoss(),
    device=device,
    fisher_path=self.fisher_path
)

if __name__ == "__main__":
    gt=GlobalTraining()
    gt.generate_split_writers(seed=42)
    nist_dataset=NistDataset()
    train_loader, eval_loader, _ = nist_dataset.build_dataset(gt.global_writers,train_rate=0.6, eval_rate=0.4, batch_size=64,seed=42)
    _, _, test_loader = nist_dataset.build_dataset(gt.global_writers,train_rate=0.0, eval_rate=0.0, batch_size=64,seed=42)
    gt.train(trn_loader=train_loader,evl_loader=eval_loader,epochs=100,load_model=False)
    gt.generate_fisher(train_loader)
    gt.generate_selected_writers(seed=42)
    gt.evaluate(tst_loader=test_loader)
    gt.save_metrics_plots()
    gt.get_outliers(force_generate=False)
    gt.generate_selected_writers(seed=42)
    gt.outliers_training(batch_size=64,epochs=100)



