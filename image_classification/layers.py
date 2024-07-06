import torch
import torch.nn as nn

from collections import OrderedDict
from utils import build_activation, make_divisible, get_same_padding, min_divisible_value

__all__ = [
    "ConvLayer",
    "IdentityLayer",
    "LinearLayer",
    "ResidualBlock",
    "ResNetBottleneckBlock",
]
CHANNEL_DIVISIBLE = 8


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_bn=True,
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.use_bn = use_bn
        self.act_func = act_func

        padding = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    groups=min_divisible_value(self.in_channels, self.groups),
                    bias=self.bias,
                )

        # batch norm
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        # activation
        self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class IdentityLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    def forward(self, x):
        return self.linear(x)


class ResidualBlock(nn.Module):
    def __init__(self, conv, shortcut):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.conv is None:
            res = x
        elif self.shortcut is None:
            res = self.conv(x)
        else:
            res = self.conv(x) + self.shortcut(x)
        return res


class ResNetBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=0.25,
        mid_channels=None,
        act_func="relu",
        groups=1,
        downsample_mode="avgpool_conv",
    ):
        super(ResNetBottleneckBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.groups = groups

        self.downsample_mode = downsample_mode

        if self.mid_channels is None:
            feature_dim = round(self.out_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        feature_dim = make_divisible(feature_dim, CHANNEL_DIVISIBLE)
        self.mid_channels = feature_dim

        # build modules
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False),
                    ),
                    ("bn", nn.BatchNorm2d(feature_dim)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        pad = get_same_padding(self.kernel_size)
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            feature_dim,
                            feature_dim,
                            kernel_size,
                            stride,
                            pad,
                            groups=groups,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(feature_dim)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        self.conv3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(feature_dim, self.out_channels, 1, 1, 0, bias=False),
                    ),
                    ("bn", nn.BatchNorm2d(self.out_channels)),
                ]
            )
        )

        if stride == 1 and in_channels == out_channels:
            self.downsample = IdentityLayer(in_channels, out_channels)
        elif self.downsample_mode == "conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                in_channels, out_channels, 1, stride, 0, bias=False
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(out_channels)),
                    ]
                )
            )
        elif self.downsample_mode == "avgpool_conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg_pool",
                            nn.AvgPool2d(
                                kernel_size=stride,
                                stride=stride,
                                padding=0,
                                ceil_mode=True,
                            ),
                        ),
                        (
                            "conv",
                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                        ),
                        ("bn", nn.BatchNorm2d(out_channels)),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x
