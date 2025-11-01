import copy
import os
from pathlib import Path

import torch

import torch.nn.functional as F
from tqdm import tqdm

from federated_adaptive_learning_nist.constants import BEST_EWC_LAMBDA
from src.federated_adaptive_learning_nist.data_utils import NistPath


from src.federated_adaptive_learning_nist.nist_logger import NistLogger
from src.federated_adaptive_learning_nist.trainers.base_trainer import BaseTrainer

project_results_dir=Path(__file__).resolve().parents[3]/"results"

def load_fisher_and_params(fisher_path, device):

    if not fisher_path:
        data_path = NistPath()
        fisher_path = project_results_dir/ f"global_results" / "fisher"
    fisher = torch.load(os.path.join(fisher_path, "fisher.pt"), map_location=device)
    global_params = torch.load(os.path.join(fisher_path, "global_params.pt"), map_location=device)
    return fisher, global_params


class EWCTrainer(BaseTrainer):
    def __init__(self, fisher_path=None, ewc_lambda=BEST_EWC_LAMBDA ):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        # print(f"device ewc: {self.device}")


        self.fisher, self.global_params = load_fisher_and_params(fisher_path, self.device)

    def ewc_loss(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels)
        ewc_penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ewc_penalty += (self.fisher[name] * (param - self.global_params[name]) ** 2).sum()

        scaling_factor = loss.item() / (ewc_penalty.item() + 1e-8)
        lambda_scaled = min(scaling_factor, self.ewc_lambda)
        return loss + (lambda_scaled / 2.0) * ewc_penalty

    def train(self, train_loader, eval_loader, epochs=10,patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"[EWC-Train] Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.ewc_loss(outputs, labels)

                if torch.isnan(loss):
                    NistLogger.error("Loss is NaN, aborting.")
                    raise ValueError("Loss is NaN")

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            self.train_accuracies.append(train_acc)

            # === Validation ===
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in eval_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            self.val_accuracies.append(val_acc)

            NistLogger.info(f"Epoch {epoch + 1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc, bad = val_acc, 0
                best_model = copy.deepcopy(self.model)
                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")
            else:
                bad += 1
                if bad >= patience:
                    NistLogger.info(f"Early stopping (patience={patience}).")
                    break

        if best_model:
            self.model = best_model
