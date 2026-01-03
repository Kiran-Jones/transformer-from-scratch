


import numpy as np


class LayerNorm:

    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps

        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)


    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta
