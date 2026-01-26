import numpy as np
from .mha import MultiHeadedAttention
from .ff_network import FeedForward
from .layer_norm import LayerNorm
from .dropout import Dropout

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

        self.dropout1 = Dropout() 
        self.dropout2 = Dropout() 
        self.dropout3 = Dropout() 

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, training=True, return_weights=False):
        self_attn_out = self.self_attn.forward(x, x, x, mask=tgt_mask)
        self_attn_out = self.dropout1.forward(self_attn_out, training)
        x = self.ln1.forward(x + self_attn_out)

        cross_attn_out = self.cross_attn.forward(x, encoder_output, encoder_output, mask=src_mask, return_weights=return_weights)
        if return_weights:
            self.cross_attn_weights = self.cross_attn.attn_weights
        cross_attn_out = self.dropout2.forward(cross_attn_out, training)
        x = self.ln2.forward(x + cross_attn_out)

        ff_out = self.ff.forward(x)
        ff_out = self.dropout3.forward(ff_out, training)
        x = self.ln3.forward(x + ff_out)

        return x
    
    def backward(self, grad_output):
        grad_norm3 = self.ln3.backward(grad_output)

        grad_ff_path = grad_norm3
        grad_bypass_ff = grad_norm3

        grad_ff_out = self.dropout3.backward(grad_ff_path)
        grad_ff = self.ff.backward(grad_ff_out)

        grad_x_after_norm2 = grad_ff + grad_bypass_ff

        grad_norm2 = self.ln2.backward(grad_x_after_norm2)

        grad_cross_path = grad_norm2
        grad_bypass_cross = grad_norm2

        grad_cross_out = self.dropout2.backward(grad_cross_path)

        dq_cross, dk_cross, dv_cross = self.cross_attn.backward(grad_cross_out)

        grad_x_after_norm1 = dq_cross + grad_bypass_cross
        
        grad_for_encoder = dk_cross + dv_cross

        grad_norm1 = self.ln1.backward(grad_x_after_norm1)

        grad_self_path = grad_norm1
        grad_bypass_self = grad_norm1

        grad_self_out = self.dropout1.backward(grad_self_path)

        dq_self, dk_self, dv_self = self.self_attn.backward(grad_self_out)

        grad_input = dq_self + dk_self + dv_self + grad_bypass_self

        return grad_input, grad_for_encoder
