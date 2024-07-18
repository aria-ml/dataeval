import torch.nn as nn


class Conv(nn.Module):
    """
    Wrapper for conv modules, so we don't have to specify everything every time
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        k=1,
        s=1,
        p=0,
        activation="relu",
        norm="instance",
    ):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        norm = self.get_norm_func(norm=norm, out_channels=out_channels)
        act = self.get_activation_func(activation=activation)
        self.module = nn.Sequential(conv, norm, act)

    def get_norm_func(self, norm: str, out_channels) -> nn.Module:
        if norm == "batch":
            return nn.BatchNorm2d(out_channels)
        if norm == "instance":
            return nn.InstanceNorm2d(out_channels)
        if norm == "layer":
            return nn.LayerNorm(out_channels)
        return nn.Identity()

    def get_activation_func(self, activation: str) -> nn.Module:
        if activation == "selu":
            return nn.SELU()
        if activation == "relu":
            return nn.ReLU()
        if activation == "leaky":
            return nn.LeakyReLU()
        if activation == "tanh":
            return nn.Tanh()
        return nn.Identity()

    def forward(self, x):
        return self.module(x)
