import numpy as np
from .embedding import Embedding
from .encoder import Encoder
from .decoder import Decoder
from .linear import Linear
from .dropout import Dropout
from .pe import positional_encoding
from .masks import create_padding_mask, create_look_ahead_mask

def create_padding_mask(seq, pad_id=0):
    # 1 = keep, 0 = mask (pad)
    seq = seq != pad_id
    return seq[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    # 1 = keep (lower triangle), 0 = mask (future positions)
    mask = np.tril(np.ones((size, size)))
    return mask 

class Transformer:
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, pad_id=0):
        self.d_model = d_model
        self.pad_id = pad_id
        
        self.embedding = Embedding(vocab_size, d_model)
        
        self.pe_func = positional_encoding
        
        self.encoder_dropout = Dropout()
        self.decoder_dropout = Dropout()
        
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        
        self.output_projection = Linear(d_model, vocab_size)
        
        
    def forward(self, src_tokens, tgt_tokens, training=True, mask=None):
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        
        # Mask
        src_mask = create_padding_mask(src_tokens, pad_id=self.pad_id)
        
        tgt_len = tgt_tokens.shape[1]
        look_ahead = create_look_ahead_mask(tgt_len)
        tgt_pad = create_padding_mask(tgt_tokens, pad_id=self.pad_id)
        tgt_mask = tgt_pad * look_ahead
        
        # Encoder
        src_emb = self.embedding.forward(src_tokens) * np.sqrt(self.d_model)
        
        self.src_embedded = src_emb + self.pe_func(src_tokens.shape[1], self.d_model)
        
        encoder_input = self.encoder_dropout.forward(self.src_embedded, training)
        
        encoder_output = self.encoder.forward(encoder_input, src_mask, training)
        
        # Decoder
        tgt_emb = self.embedding.forward(tgt_tokens) * np.sqrt(self.d_model)
        
        self.tgt_embedded = tgt_emb + self.pe_func(tgt_tokens.shape[1], self.d_model)
        
        decoder_input = self.decoder_dropout.forward(self.tgt_embedded, training)
        
        decoder_output = self.decoder.forward(decoder_input, encoder_output, src_mask, tgt_mask, training)
        
        logits = self.output_projection.forward(decoder_output)
        
        return logits

    def backward(self, grad_logits):
        """
        Orchestrates the full backprop from Logits -> Embeddings
        """
        grad_decoder_output = self.output_projection.backward(grad_logits)
        
        grad_decoder_input, grad_encoder_output = self.decoder.backward(grad_decoder_output)
        
        grad_tgt_embedded = self.decoder_dropout.backward(grad_decoder_input)

        self.embedding.backward(grad_tgt_embedded, token_ids=self.tgt_tokens)
        
        grad_encoder_input = self.encoder.backward(grad_encoder_output)
        
        grad_src_embedded = self.encoder_dropout.backward(grad_encoder_input)
        
        self.embedding.backward(grad_src_embedded, token_ids=self.src_tokens)
        
        return None
    
    def make_src_mask(self, src):
        return create_padding_mask(src, pad_id=self.pad_id)
    
    def make_tgt_mask(self, tgt):
        look_ahead_mask = create_look_ahead_mask(tgt.shape[1])
        dec_target_padding_mask = create_padding_mask(tgt, pad_id=self.pad_id)
        combined_mask = dec_target_padding_mask * look_ahead_mask
        return combined_mask
