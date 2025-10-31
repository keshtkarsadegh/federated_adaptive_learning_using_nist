import json
import torch

from torch import nn
from matplotlib import pyplot as plt
from src.federated_adaptive_learning_nist.model import  FlexibleCNN
from src.federated_adaptive_learning_nist.nist_logger import NistLogger
from src.federated_adaptive_learning_nist.outliers.oulier_training import outliers_train
from src.federated_adaptive_learning_nist.outliers.outliers_statics import get_low_acc_writers
from src.federated_adaptive_learning_nist.outliers.selecting_outliers import select_outliers
from src.federated_adaptive_learning_nist.trainers.base_trainer import BaseTrainer
from src.federated_adaptive_learning_nist.trainers.trainer_utils import compute_and_save_fisher_and_params
from src.federated_adaptive_learning_nist.data_utils import split_local_global_writers, NistDataset, NistPath
from pathlib import Path
project_results_dir=Path(__file__).resolve().parents[3]/"results"

class GlobalTraining:
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
        (project_results_dir).mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        (project_results_dir/"outliers").mkdir(parents=True, exist_ok=True)
        self.outliers_result_path.mkdir(parents=True, exist_ok=True)

    def train(self,trn_loader,evl_loader,epochs,load_model=False):
        if load_model:
            NistLogger.info(f"Loading model from {self.model_path}")
            self.model =self.trainer.load_model(self.model_path)
        else:

            self.trainer.set_model(self.model)
            self.trainer.train(train_loader=trn_loader, eval_loader=evl_loader, epochs=epochs)
            self.model=self.trainer.get_model()
            torch.save(self.model.state_dict(), self.model_path)
    def evaluate(self,tst_loader):
        self.model.load_state_dict(torch.load(self.model_path))
        self.trainer.set_model(self.model)
        self.trainer.evaluate(tst_loader)

    def generate_split_writers(self,global_size=0.03,seed=42):
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
        plt.axhline(y=self.trainer.test_acc, color='r', linestyle='--', label=f"Test Accuracy: {self.trainer.test_acc:.4f}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training / Validation Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_path/f"{self.name}_plots.png")
        NistLogger.info(f"Saved accuracy plot to {self.results_path/f"{self.name}_plots.png"}")
    def generate_selected_writers(self,seed=42):
        self.selected_writers= get_low_acc_writers(self.clients_acc_on_global_path,self.selected_outliers_path,self.outliers_result_path,seed=seed)
        return self.selected_writers
    def get_outliers(self,force_generate=False):
        select_outliers(self.trainer.model, self.local_writers, self.clients_acc_on_global_path,force_generate=force_generate)

    def outliers_training(self,batch_size,epochs):
        outliers_train(self.global_writers, self.selected_writers, batch_size=batch_size, num_epochs=epochs, outliers_results_path=self.outliers_result_path)
    def generate_fisher(self,trn_loader):
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



