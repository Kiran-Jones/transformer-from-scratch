import numpy as np
from .encoder_layer import EncoderLayer
from .layer_norm import LayerNorm

class Encoder:
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        self.d_model = d_model
        self.num_layers = num_layers
        
        
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_heads, d_ff))
            
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, mask=None, training=True):
        for layer in self.layers:
            x = layer.forward(x, mask, training)
            
        return self.norm.forward(x)
    
    def backward(self, grad_output):
        grad = self.norm.backward(grad_output)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return grad
