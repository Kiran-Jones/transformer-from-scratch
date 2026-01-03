

import numpy as np



class Linear:

    def __init__(self, d_model, vocab_size):
        self.vocab_size = vocab_size
        self.d_model = d_model

        limit = np.sqrt(6 / (d_model + vocab_size))
        self.weight = np.random.uniform(-limit, limit, (vocab_size, d_model))



    def forward(self, x):
        return x @ self.weight.T