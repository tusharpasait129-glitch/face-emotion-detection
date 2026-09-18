"""
EmotionCNN: a compact convolutional network for 7-class facial expression
classification on 48x48 grayscale crops (the FER2013 format).

Design notes
------------
The network is intentionally small rather than borrowing a large backbone
like ResNet: FER2013 has only ~28k training images at 48x48 resolution, and a
deep, heavily-parameterised network overfits that data quickly. Three
conv blocks (each conv -> batchnorm -> ReLU -> maxpool) progressively halve
the spatial size while increasing channel depth, followed by two fully
connected layers with dropout for regularisation.

Input:  (batch, 1, 48, 48)
Output: (batch, 7) raw logits (use softmax / argmax outside the model)
"""

import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 48x48 -> 24x24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 24x24 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 12x12 -> 6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through the model and print the
    # output shape and parameter count.
    model = EmotionCNN()
    dummy = torch.randn(4, 1, 48, 48)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Trainable parameters: {n_params:,}")
