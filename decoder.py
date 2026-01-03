
from decoder_layer import DecoderLayer
import numpy as np


class Decoder:

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.layers = []

        for _ in range(num_layers):
            self.layers.append(DecoderLayer(d_model, num_heads, d_ff))

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        if tgt_mask is None:
            seq_len = x.shape[1]
            tgt_mask = np.tril(np.ones((seq_len, seq_len)))

        for layer in self.layers:
            x = layer.forward(x, encoder_output, src_mask, tgt_mask)
        return x
    