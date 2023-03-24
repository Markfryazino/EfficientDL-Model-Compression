import torch
import torchvision
import torch.nn as nn

from torch import Tensor

from typing import Optional, Callable


def get_big_resnet():
    model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048, 10)
    return model


def get_small_resnet() -> torch.nn.Module:
    model = torchvision.models.resnet101(weights=None)
    model.fc = torch.nn.Linear(2048, 10)
    model.layer3 = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    return model


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class QuantizableBottleneck(nn.Module):
    def __init__(
        self,
        bottleneck
    ) -> None:
        super().__init__()
        self.conv1 = bottleneck.conv1
        self.bn1 = bottleneck.bn1
        self.conv2 = bottleneck.conv2
        self.bn2 = bottleneck.bn2
        self.conv3 = bottleneck.conv3
        self.bn3 = bottleneck.bn3
        self.relu = bottleneck.relu
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


def replace_bottleneck_with_quantizable_bottleneck(model):
    for name, module in model.named_children():
        if isinstance(module, torchvision.models.resnet.Bottleneck):
            quantizable_bottleneck = QuantizableBottleneck(module)
            setattr(model, name, quantizable_bottleneck)
        else:
            replace_bottleneck_with_quantizable_bottleneck(module)
