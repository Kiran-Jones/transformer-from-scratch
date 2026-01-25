

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

        self.dropout1 = Dropout()
        self.dropout2 = Dropout()       

    def forward(self, x, mask=None, training=True):
        
        attn_out = self.self_attn.forward(x, x, x, mask)
        attn_out = self.dropout1.forward(attn_out, training)
        x_after_attn = self.ln1.forward(x + attn_out)

        ff_out = self.ff.forward(x_after_attn)
        ff_out = self.dropout2.forward(ff_out, training)

        x_out = self.ln2.forward(x_after_attn + ff_out)
        return x_out
    
    def backward(self, grad_output):
        grad_norm2 = self.ln2.backward(grad_output)

        grad_ff_path = grad_norm2
        grad_bypass_ff = grad_norm2

        grad_ff_out = self.dropout2.backward(grad_ff_path)
        grad_ff = self.ff.backward(grad_ff_out)

        grad_x_after_attn = grad_ff + grad_bypass_ff

        grad_norm1 = self.ln1.backward(grad_x_after_attn)
        
        grad_mha_path = grad_norm1
        grad_bypass_mha = grad_norm1

        grad_mha_out = self.dropout1.backward(grad_mha_path)

        dq, dk, dv = self.self_attn.backward(grad_mha_out)

        grad_input = dq + dk + dv + grad_bypass_mha

        return grad_input
