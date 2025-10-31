import copy
import torch

from torch import nn, optim
from tqdm import tqdm

from src.federated_adaptive_learning_nist.model import FlexibleCNN
from src.federated_adaptive_learning_nist.nist_logger import NistLogger
"""
base_trainer.py

Purpose:
    Provides a generic trainer class for federated adaptive learning experiments
    (used as a base for EWC, Distillation, Logit Consistency, Prox, etc.).

Capabilities:
    - Initialize and manage a CNN model for NIST digit classification.
    - Train a client model with standard or custom loss functions.
    - Track per-epoch training and validation accuracy.
    - Select and retain the best validation model.
    - Evaluate trained models on test datasets.
    - Reload saved model weights.

Usage:
    trainer = BaseTrainer()
    trainer.train(train_loader, eval_loader, epochs=10)
    acc = trainer.evaluate(test_loader)
    model = trainer.get_model()
"""


class BaseTrainer:
    """
        Generic trainer for federated adaptive learning on NIST.

        Responsibilities:
            - Construct and manage a CNN model (FlexibleCNN).
            - Perform client-side training with Adam optimizer.
            - Track and log accuracy metrics.
            - Evaluate models on unseen test sets.
            - Provide hooks for specialized trainers (EWC, distillation, etc.)
              by supporting `custom_loss`.

        Attributes:
            device (str): Training device ('cuda' if available, else 'cpu').
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for optimizer.
            model (FlexibleCNN): Active CNN model.
            optimizer (torch.optim.Optimizer): Adam optimizer.
            criterion (nn.Module): Default loss function (CrossEntropy).
            custom_loss (callable, optional): Optional override for loss.
            test_acc (float): Last computed test accuracy.
            train_accuracies (list): History of training accuracies per epoch.
            val_accuracies (list): History of validation accuracies per epoch.
        """
    def __init__(self, learning_rate=1e-3, weight_decay=1e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"device base trainer: {self.device}")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = FlexibleCNN(num_classes=10, complexity=3, dropout_rate=0.1, use_dropout=True, regularization=False)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.custom_loss = None

        self.test_acc = None
        self.metrics = None
        self.train_accuracies = []
        self.val_accuracies = []

    def set_model(self, model):
        """
            Replace internal model with a copy of the provided one.

            Args:
                model (torch.nn.Module): Model to adopt and train.
            """
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def load_model(self, model_path):
        """
            Load model weights from a saved file.

            Args:
                model_path (str or Path): Path to model checkpoint.

            Returns:
                torch.nn.Module: The loaded model.
            """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return self.model

    def get_model(self):
        """
           Return the current managed model.

           Returns:
               torch.nn.Module: Active CNN model.
           """
        return self.model

    def train(self, train_loader, eval_loader, epochs=10):
        """
            Train the model for a number of epochs.

            Args:
                train_loader (DataLoader): Training data loader.
                eval_loader (DataLoader): Validation data loader.
                epochs (int, optional): Number of epochs to train. Default=10.

            Process:
                - Runs training and validation loops.
                - Tracks accuracy history.
                - Selects best model by validation accuracy.
                - Restores best model at the end.
            """
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model = None
        best_val_acc = -float('inf')

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"[Training] Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.custom_loss(outputs, labels) if self.custom_loss else self.criterion(outputs, labels)

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
                best_val_acc = val_acc
                best_model = copy.deepcopy(self.model)
                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")

        if best_model:
            self.model = best_model

    def evaluate(self, test_loader):
        """
            Evaluate current model on a test dataset.

            Args:
                test_loader (DataLoader): Test data loader.

            Returns:
                float: Test accuracy.
            """
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        self.test_acc = correct / total
        NistLogger.info(f"Test Accuracy: {self.test_acc:.4f}")
        return self.test_acc