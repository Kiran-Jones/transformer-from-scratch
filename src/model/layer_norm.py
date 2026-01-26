


import numpy as np


class LayerNorm:

    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps

        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)


    def forward(self, x):
        self.x = x
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(variance + self.eps)
        self.x_norm = (x - mean) * self.std_inv       
        return self.gamma * self.x_norm + self.beta    
    
    
    def backward(self, grad_output):
        self.grad_beta = grad_output.sum(axis=(0, 1))
        self.grad_gamma = (grad_output * self.x_norm).sum(axis=(0, 1))

        N = grad_output.shape[-1]
        dx_norm = grad_output * self.gamma
        term1 = N * dx_norm
        term2 = dx_norm.sum(axis=-1, keepdims=True)
        term3 = self.x_norm * (dx_norm * self.x_norm).sum(axis=-1, keepdims=True)

        grad_input = (1.0 / N) * self.std_inv * (term1 - term2 - term3)

        return grad_input


