#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/layers/activations.py
# Create: 2020-05-04 10:33:14
# LastAuthor: Shihua Liang
# lastTime: 2020-05-04 10:33:52
# --------------------------------------------------------

import math

import torch
from torch import nn

activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ReLU6": nn.ReLU6,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    "SiLU": nn.SiLU,
    "HardSwish": nn.Hardswish,
    "Hardswish": nn.Hardswish,
    None: nn.Identity,
}


def act_layers(name):
    assert name in activations.keys()
    if name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    else:
        return activations[name](inplace=True)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
