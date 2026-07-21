"""
Simple CNN for Cat vs Dog classification.
"""

import torch
import torch.nn as nn

from models.layers import ConvBlock


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = SimpleCNN()

    dummy = torch.randn(8, 3, 128, 128)

    output = model(dummy)

    print(model)

    print()

    print("Input Shape :", dummy.shape)

    print("Output Shape:", output.shape)
