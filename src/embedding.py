

import numpy as np

class Embedding:

    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weight = np.random.randn(vocab_size, d_model) * 0.01
        self.grad_weights = None


    def forward(self, token_ids):
        self.token_ids = token_ids
        return self.weight[token_ids]
    

    def backward(self, grad_output, token_ids=None):
        if token_ids is None:
            token_ids = self.token_ids
        if self.grad_weights is None:
            self.grad_weights = np.zeros_like(self.weight)
        flat_ids = token_ids.flatten()
        flat_grads = grad_output.reshape(-1, grad_output.shape[-1])
        np.add.at(self.grad_weights, flat_ids, flat_grads)
        return None
