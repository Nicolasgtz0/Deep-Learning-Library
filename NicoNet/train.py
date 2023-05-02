#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:36:26 2023

@author: nicolasgutierrez
"""
#Credits: Joel Grus

"""
Here's a function that can train a neural net
"""

from NicoNet.tensor import Tensor
from NicoNet.nn import NeuralNet
from NicoNet.loss import Loss, MSE
from NicoNet.optim import Optimizer, SGD
from NicoNet.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs) 
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)