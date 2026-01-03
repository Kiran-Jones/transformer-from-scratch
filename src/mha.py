
from sdpa import sdpa
import numpy as np

class MultiHeadedAttention:

    def __init__(self, d_model=512, num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len_q, _ = Q.shape
        _, seq_len_k, _ = K.shape
        
        Q_proj = Q @ self.W_q
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v
        
        Q_heads = Q_proj.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # apply scaled dot-product attention with mask
        attn = sdpa(Q_heads, K_heads, V_heads, mask=mask)
        
        # concatenate heads
        output = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        
        return output @ self.W_o
        