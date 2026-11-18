import torch
from torch import nn


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.padding, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation):
        super().__init__()
        self.net = nn.Sequential(CausalConv(inputs, outputs, kernel, dilation), nn.ReLU(),
                                 CausalConv(outputs, outputs, kernel, dilation), nn.ReLU())
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class TCN(nn.Module):
    def __init__(self, inputs, channels=(32, 32, 64, 64, 64, 64, 64), kernel=3):
        super().__init__()
        blocks = []
        self.receptive_field = 1
        for level, output in enumerate(channels):
            dilation = 2 ** level
            blocks.append(ResidualBlock(inputs, output, kernel, dilation))
            inputs = output
            self.receptive_field += 2 * (kernel - 1) * dilation
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv1d(inputs, 1, 1)

    def timestep_logits(self, x):
        return self.head(self.backbone(x.transpose(1, 2))).squeeze(1)

    def forward(self, x):
        # Decision at the final observed sample; no pooling over future context.
        return self.timestep_logits(x)[:, -1]
