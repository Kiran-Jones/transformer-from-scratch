
import numpy as np


class FeedForward:

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        self.W_1 = np.random.randn(d_model, d_ff)
        self.b_1 = np.random.randn(d_ff, )

        self.W_2 = np.random.randn(d_ff, d_model)
        self.b_2 = np.random.randn(d_model, )

        # self.W_1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        # self.W_2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)

    def forward(self, x):
        return np.maximum(0, x @ self.W_1 + self.b_1) @ self.W_2 + self.b_2
