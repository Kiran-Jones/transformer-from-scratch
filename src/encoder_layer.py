

from mha import MultiHeadedAttention
from ff_network import FeedForward
from layer_norm import LayerNorm
from dropout import Dropout

import numpy as np



class EncoderLayer:

    def __init__(self, d_model=512, num_heads=8, d_ff=2048):

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        self.dropout = Dropout()
        
    def forward(self, x, mask=None):
        attn_out = self.self_attn.forward(x, x, x)
        attn_out = self.dropout.forward(attn_out)
        x = self.ln1.forward(x + attn_out)

        ff_out = self.ff.forward(x)
        ff_out = self.dropout.forward(ff_out)
        x = self.ln2.forward(x + ff_out)
        return x
