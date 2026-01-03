

from mha import MultiHeadedAttention
from ff_network import FeedForward
from layer_norm import LayerNorm

import numpy as np



class DecoderLayer:

    def __init__(self, d_model=512, num_heads=8, d_ff=2048):

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.cross_attn = MultiHeadedAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.ln1.forward(x + self.self_attn.forward(x, x, x, mask=tgt_mask))
        x = self.ln2.forward(x + self.cross_attn.forward(x, encoder_output, encoder_output, mask=src_mask))
        return self.ln3.forward(x + self.ff.forward(x))
