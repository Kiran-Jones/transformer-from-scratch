

import numpy as np

class Embedding:

    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weight = np.random.randn(vocab_size, d_model) * 0.01


    def forward(self, token_ids):
        return self.weight[token_ids]