#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:23:50 2023

@author: nicolasgutierrez
"""
#Credits: Joel Grus

"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropagation
"""
from NicoNet.nn import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer): #Stochastic Gradient Descent 
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad