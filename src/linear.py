

import numpy as np



class Linear:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim    

        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weight = np.random.uniform(-limit, limit, (input_dim, output_dim))

        self.bias = np.zeros(output_dim)


    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias    

    def backward(self, grad_output):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])

        self.grad_weights = np.dot(x_flat.T, grad_output_flat)
        self.grad_bias = grad_output_flat.sum(axis=0)
        grad_input = np.dot(grad_output, self.weight.T).reshape(self.x.shape)

        return grad_input