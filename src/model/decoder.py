import numpy as np
from .decoder_layer import DecoderLayer
from .layer_norm import LayerNorm

class Decoder:
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        self.d_model = d_model
        self.num_layers = num_layers
                
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(d_model, num_heads, d_ff))
            
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask, training=True, return_weights=False):
        for layer in self.layers:
            x = layer.forward(x, encoder_output, src_mask, tgt_mask, training, return_weights=return_weights)

        return self.norm.forward(x)
    
    def backward(self, grad_output):
        grad = self.norm.backward(grad_output)
        
        total_grad_for_encoder = 0
        
        for layer in reversed(self.layers):
            grad, grad_enc = layer.backward(grad)
            
            total_grad_for_encoder = total_grad_for_encoder + grad_enc
            
        return grad, total_grad_for_encoder
