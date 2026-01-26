import numpy as np
from .linear import Linear

class FeedForward:

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        hidden = self.linear1.forward(x)
        
        self.relu_input = hidden
        hidden = np.maximum(0, hidden)
        
        output = self.linear2.forward(hidden)
        
        return output
    
    def backward(self, grad_output):
        grad_hidden = self.linear2.backward(grad_output)

        relu_mask = (self.relu_input > 0).astype(float)
        grad_relu = grad_hidden * relu_mask

        grad_input = self.linear1.backward(grad_relu)

        return grad_input