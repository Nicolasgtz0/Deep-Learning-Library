#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:19:08 2023

@author: nicolasgutierrez
"""
#Credits: Joel Grus

"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself, although
we're not going to make it one.
"""
from typing import Sequence, Iterator, Tuple

from NicoNet.tensor import Tensor
from NicoNet.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
