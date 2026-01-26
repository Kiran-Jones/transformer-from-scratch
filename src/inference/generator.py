
import numpy as np
from model.pe import positional_encoding
from model.masks import create_padding_mask, create_look_ahead_mask
from model.utils import softmax

class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _encode_source(self, src_text):
        """Encode source text and run through encoder."""
        sos_id = self.tokenizer.encoder['<SOS>']
        eos_id = self.tokenizer.encoder['<EOS>']

        encoded_src = self.tokenizer.encode(src_text)
        encoded_src = [sos_id] + encoded_src + [eos_id]
        src_seq = np.array(encoded_src).reshape(1, -1)

        src_mask = self.model.make_src_mask(src_seq)

        src_emb = self.model.embedding.forward(src_seq) * np.sqrt(self.model.d_model)
        src_emb += positional_encoding(src_seq.shape[1], self.model.d_model)
        encoder_input = self.model.encoder_dropout.forward(src_emb, training=False)

        encoder_output = self.model.encoder.forward(encoder_input, src_mask, training=False)

        return encoder_output, src_mask, sos_id, eos_id

    def _decode_step(self, tgt_seq, encoder_output, src_mask, return_weights=False):
        """Run one decoding step and return logits."""
        tgt_input = np.array(tgt_seq).reshape(1, -1)

        look_ahead_mask = create_look_ahead_mask(tgt_input.shape[1])
        dec_target_padding_mask = create_padding_mask(tgt_input, pad_id=self.model.pad_id)
        tgt_mask = np.maximum(dec_target_padding_mask, look_ahead_mask)

        tgt_emb = self.model.embedding.forward(tgt_input) * np.sqrt(self.model.d_model)
        tgt_emb += positional_encoding(tgt_input.shape[1], self.model.d_model)
        decoder_input = self.model.decoder_dropout.forward(tgt_emb, training=False)

        decoder_output = self.model.decoder.forward(
            decoder_input,
            encoder_output,
            src_mask,
            tgt_mask,
            training=False,
            return_weights=return_weights
        )

        logits = self.model.output_projection.forward(decoder_output)
        return logits

    def generate(self, src_text, max_len=50, min_len=5, beam_width=4):
        """Generate translation using beam search decoding."""
        encoder_output, src_mask, sos_id, eos_id = self._encode_source(src_text)

        # Initialize beams: list of (sequence, cumulative_log_prob)
        beams = [([sos_id], 0.0)]
        completed = []

        for step in range(max_len):
            all_candidates = []

            for seq, score in beams:
                # Skip completed sequences
                if seq[-1] == eos_id:
                    completed.append((seq, score))
                    continue

                logits = self._decode_step(seq, encoder_output, src_mask)
                log_probs = np.log(softmax(logits[0, -1, :]) + 1e-10)

                # Block EOS before min_len
                if step < min_len:
                    log_probs[eos_id] = -1e9

                # Get top beam_width candidates for this beam
                top_indices = np.argsort(log_probs)[-beam_width:]

                for idx in top_indices:
                    new_seq = seq + [int(idx)]
                    new_score = score + log_probs[idx]
                    all_candidates.append((new_seq, new_score))

            if not all_candidates:
                break

            # Select top beam_width candidates overall
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # Early exit if all beams have ended
            if all(seq[-1] == eos_id for seq, _ in beams):
                completed.extend(beams)
                break

        # Add any remaining beams to completed
        completed.extend(beams)

        # Select best sequence with length normalization
        def normalized_score(seq, score):
            length = len(seq) - 1  # Exclude SOS
            return score / (length ** 0.6) if length > 0 else score

        best_seq, _ = max(completed, key=lambda x: normalized_score(x[0], x[1]))

        # Remove SOS and EOS
        if best_seq and best_seq[0] == sos_id:
            best_seq = best_seq[1:]
        if best_seq and best_seq[-1] == eos_id:
            best_seq = best_seq[:-1]

        return self.tokenizer.decode(best_seq)

    def _collect_attention_weights(self):
        """Collect cross-attention weights from all decoder layers."""
        step_attention = []
        for layer in self.model.decoder.layers:
            step_attention.append(layer.cross_attn_weights.copy())
        # Shape: (num_layers, num_heads, src_len)
        token_attention = np.stack(step_attention)[:, 0, :, -1, :]
        return token_attention

    def generate_with_attention(self, src_text, max_len=50, min_len=5, beam_width=4):
        """Generate translation using beam search and collect attention weights for the best beam."""
        encoder_output, src_mask, sos_id, eos_id = self._encode_source(src_text)

        # Get source tokens for visualization
        src_tokens = ['<SOS>']
        for tok_id in self.tokenizer.encode(src_text):
            src_tokens.append(self.tokenizer.decode([tok_id]))
        src_tokens.append('<EOS>')

        num_layers = len(self.model.decoder.layers)
        num_heads = self.model.decoder.layers[0].cross_attn.num_heads

        # Initialize beams: (sequence, score, attention_history)
        beams = [([sos_id], 0.0, [])]
        completed = []

        for step in range(max_len):
            all_candidates = []

            for seq, score, attn_history in beams:
                if seq[-1] == eos_id:
                    completed.append((seq, score, attn_history))
                    continue

                # Decode with attention collection
                logits = self._decode_step(seq, encoder_output, src_mask, return_weights=True)
                token_attention = self._collect_attention_weights()
                log_probs = np.log(softmax(logits[0, -1, :]) + 1e-10)

                if step < min_len:
                    log_probs[eos_id] = -1e9

                top_indices = np.argsort(log_probs)[-beam_width:]

                for idx in top_indices:
                    new_seq = seq + [int(idx)]
                    new_score = score + log_probs[idx]
                    new_attn = attn_history + [token_attention]
                    all_candidates.append((new_seq, new_score, new_attn))

            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            if all(seq[-1] == eos_id for seq, _, _ in beams):
                completed.extend(beams)
                break

        completed.extend(beams)

        def normalized_score(seq, score):
            length = len(seq) - 1
            return score / (length ** 0.6) if length > 0 else score

        best_seq, _, best_attn = max(completed, key=lambda x: normalized_score(x[0], x[1]))

        # Get target tokens for visualization
        tgt_tokens = []
        for tok_id in best_seq:
            if tok_id == sos_id:
                tgt_tokens.append('<SOS>')
            elif tok_id == eos_id:
                tgt_tokens.append('<EOS>')
            else:
                tgt_tokens.append(self.tokenizer.decode([tok_id]))

        # Prepare output sequence
        output_seq = best_seq[:]
        if output_seq and output_seq[0] == sos_id:
            output_seq = output_seq[1:]
        if output_seq and output_seq[-1] == eos_id:
            output_seq = output_seq[:-1]

        translation = self.tokenizer.decode(output_seq)

        # Stack attention weights: (tgt_len, num_layers, num_heads, src_len)
        if best_attn:
            attention_weights = np.stack(best_attn)
        else:
            attention_weights = np.zeros((1, num_layers, num_heads, len(src_tokens)))

        return translation, attention_weights, src_tokens, tgt_tokens, num_layers, num_heads 
