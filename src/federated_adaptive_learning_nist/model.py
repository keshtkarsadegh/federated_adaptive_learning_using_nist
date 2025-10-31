from pathlib import Path

import torch

import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        complexity=3,
        dropout_rate=0.1,
        use_dropout=True,
        regularization=None
    ):
        super(FlexibleCNN, self).__init__()

        self.features = nn.Sequential()
        input_channels = 1  # grayscale

        for i in range(complexity):
            out_channels = 32 * (i + 1)
            self.features.add_module(f"conv_{i}", nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1))
            self.features.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels))
            self.features.add_module(f"relu_{i}", nn.ReLU())
            self.features.add_module(f"pool_{i}", nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = out_channels

        # ✅ Shrink output size before FC to prevent overfitting / huge flatten size
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d((4, 4)))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)
            x = self.features(dummy)
            self.flatten_size = x.view(1, -1).shape[1]
        # NistLogger.info("FLATTEN SIZE AT RUNTIME:", self.flatten_size)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None
        self.regularization = regularization
        self._reg_lambda = 1e-5

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_regularization_loss(self):
        if self.regularization is None:
            return 0.0
        reg_loss = 0.0
        for param in self.parameters():
            if self.regularization == "l1":
                reg_loss += param.abs().sum()
            elif self.regularization == "l2":
                reg_loss += (param ** 2).sum()
            else:
                raise ValueError(f"❌ Unknown regularization type: {self.regularization}")
        return self._reg_lambda * reg_loss

    def body_parameters(self):
        return list(self.features.parameters()) + list(self.fc1.parameters())

    def head_parameters(self):
        return list(self.fc2.parameters())

    def freeze_body(self):
        for p in self.body_parameters():
            p.requires_grad = False

    def unfreeze_body(self):
        for p in self.body_parameters():
            p.requires_grad = True

    def set_dropout(self, enable=True):
        self.dropout = nn.Dropout(self.dropout.p) if enable else None

    def print_param_groups(self):
        print(f"Body Params: {sum(p.numel() for p in self.body_parameters())}")
        print(f"Head Params: {sum(p.numel() for p in self.head_parameters())}")



