#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:49:11 2023

@author: nicolasgutierrez
"""
# Credits: Joel Grus

"""
A loss function measures how good our predictions are, 
we can use this to adjust the parameteres of our network.

"""

import numpy as np 
from NicoNet.tensor import Tensor 

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    

class MSE(Loss):
    """
    MSE is mean squared error, although we're just going to do total squared error.
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)