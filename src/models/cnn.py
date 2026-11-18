"""Compact convolutional baseline using only the available historical window."""
from torch import nn


class ConvPredictor(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(inputs, 32, 9, stride=2, padding=4), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2))).squeeze(-1)
