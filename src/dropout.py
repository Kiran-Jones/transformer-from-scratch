

import numpy as np


class Dropout:

    def __init__(self, dropout=0.1):
        self.dropout = dropout
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout, size=x.shape)
            return x * self.mask / (1 - self.dropout)
        else:
            return x
        
    def backward(self, grad_output):
        return grad_output * self.mask / (1 - self.dropout)

