
from encoder_layer import EncoderLayer



class Encoder:

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.layers = []

        for _ in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_heads, d_ff))
            
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x
    

        
