


from embedding import Embedding
from encoder import Encoder
from decoder import Decoder
from linear import Linear

from pe import positional_encoding
from utils import softmax
import numpy as np

class Transformer:

    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, d_model)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)

        self.output_projection = Linear(d_model, vocab_size)
        self.output_projection.weight = self.embedding.weight


    def forward(self, src_tokens, tgt_tokens):
        # 1. encoder path
        src_embedding = self.embedding.forward(src_tokens)
        src_pos_enc = positional_encoding(src_tokens.shape[1], self.d_model)

        encoder_input = src_embedding * np.sqrt(self.d_model) + src_pos_enc
        encoder_output = self.encoder.forward(encoder_input)
    
        # 2. decoder path using encoder output
        tgt_embedding = self.embedding.forward(tgt_tokens)
        tgt_pos_enc = positional_encoding(tgt_tokens.shape[1], self.d_model)

        decoder_input = tgt_embedding * np.sqrt(self.d_model) + tgt_pos_enc
        decoder_output = self.decoder.forward(decoder_input, encoder_output)

        # 3. output projection
        logits = self.output_projection.forward(decoder_output)

        return logits


if __name__ == "__main__":

    # config
    vocab_size = 100
    d_model = 64 
    num_heads = 4
    d_ff = 128
    num_layers = 2

    
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    # test data
    batch_size = 2
    src_len = 5
    tgt_len = 4

    # random token IDs
    src_tokens = np.random.randint(0, vocab_size, size=(batch_size, src_len))
    tgt_tokens = np.random.randint(0, vocab_size, size=(batch_size, tgt_len))

    print("Source tokens shape:", src_tokens.shape) 
    print("Target tokens shape:", tgt_tokens.shape) 

    # forward pass
    logits = transformer.forward(src_tokens, tgt_tokens)

    print("Output logits shape:", logits.shape)  
    print("Logits range:", logits.min(), "to", logits.max())

    # sanity checks
    assert logits.shape == (batch_size, tgt_len, vocab_size), "Wrong output shape!"
    assert not np.isnan(logits).any(), "NaN in output!"
    assert not np.isinf(logits).any(), "Inf in output!"

    print("Forward pass completed!")