
from sdpa import sdpa, sdpa_backward
from linear import Linear
import numpy as np

class MultiHeadedAttention:

    def __init__(self, d_model=512, num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None, return_weights=False):
        batch_size, seq_len_q, _ = Q.shape
        _, seq_len_k, _ = K.shape

        Q_proj = self.W_q.forward(Q)
        K_proj = self.W_k.forward(K)
        V_proj = self.W_v.forward(V)

        Q_heads = Q_proj.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        self.Q_heads = Q_heads
        self.K_heads = K_heads
        self.V_heads = V_heads
        self.mask = mask

        # apply scaled dot-product attention with mask
        if return_weights:
            attn, attn_weights = sdpa(Q_heads, K_heads, V_heads, mask=mask, return_weights=True)
            self.attn_weights = attn_weights
        else:
            attn = sdpa(Q_heads, K_heads, V_heads, mask=mask)

        # concatenate heads
        output = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)

        return self.W_o.forward(output)
        

    def backward(self, grad_output):
        grad_concat = self.W_o.backward(grad_output)

        batch_size, seq_len, _ = grad_concat.shape

        grad_concat = grad_concat.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        grad_sdpa_output = grad_concat.transpose(0, 2, 1, 3)

        grad_q_heads, grad_k_heads, grad_v_heads = sdpa_backward(
            grad_sdpa_output, 
            self.Q_heads, 
            self.K_heads, 
            self.V_heads, 
            self.mask
        )

        def merge_heads(x):
            x = x.transpose(0, 2, 1, 3)
            return x.reshape(batch_size, x.shape[1], self.d_model)
        
        grad_q_merged = merge_heads(grad_q_heads)
        grad_k_merged = merge_heads(grad_k_heads)
        grad_v_merged = merge_heads(grad_v_heads)

        grad_q_input = self.W_q.backward(grad_q_merged)
        grad_k_input = self.W_k.backward(grad_k_merged)
        grad_v_input = self.W_v.backward(grad_v_merged)

        return grad_q_input, grad_k_input, grad_v_input