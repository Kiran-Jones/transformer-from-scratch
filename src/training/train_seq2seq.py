import argparse
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
from model.transformer import Transformer
from inference.tokenizer import Tokenizer
from training.optimizer import Adam

def load_parallel(filepaths):
    pairs = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                en = parts[1].strip()
                es = parts[3].strip()
                if en and es:
                    pairs.append((en, es))
    return pairs


def get_batch(pairs, tokenizer, batch_size, max_len):
    pad_id = tokenizer.encoder["<pad>"]
    sos_id = tokenizer.encoder["<SOS>"]
    eos_id = tokenizer.encoder["<EOS>"]

    idx = np.random.randint(0, len(pairs), batch_size)
    batch = [pairs[i] for i in idx]

    src_batch, dec_in_batch, dec_tgt_batch = [], [], []

    for en, es in batch:
        src_tokens = tokenizer.encode(en)
        tgt_tokens = tokenizer.encode(es)

        # Encoder: [SOS] EN [EOS] (recommended)
        src = [sos_id] + src_tokens + [eos_id]
        src = src[:max_len]
        src = src + [pad_id] * (max_len - len(src))

        # Decoder input: [SOS] ES
        dec_in = [sos_id] + tgt_tokens
        dec_in = dec_in[:max_len]
        dec_in = dec_in + [pad_id] * (max_len - len(dec_in))

        # Decoder target/labels: ES [EOS]
        dec_tgt = tgt_tokens + [eos_id]
        dec_tgt = dec_tgt[:max_len]
        dec_tgt = dec_tgt + [pad_id] * (max_len - len(dec_tgt))

        src_batch.append(src)
        dec_in_batch.append(dec_in)
        dec_tgt_batch.append(dec_tgt)

    return np.array(src_batch), np.array(dec_in_batch), np.array(dec_tgt_batch)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a seq2seq Transformer for translation")
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="Input TSV file(s) with parallel sentences")
    parser.add_argument("--vocab-encoder", default="src/data/vocab_encoder.json",
                        help="Path to vocab encoder JSON (default: src/data/vocab_encoder.json)")
    parser.add_argument("--vocab-merges", default="src/data/vocab_merges.json",
                        help="Path to vocab merges JSON (default: src/data/vocab_merges.json)")
    parser.add_argument("-o", "--output", default="models/seq2seq_model.pkl",
                        help="Output model weights path (default: models/seq2seq_model.pkl)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of training steps (default: 500)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--max-len", type=int, default=32,
                        help="Max sequence length (default: 32)")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Model dimension (default: 64)")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of encoder/decoder layers (default: 2)")
    parser.add_argument("--d-ff", type=int, default=128,
                        help="Feedforward hidden dimension (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing model to continue training from")
    return parser.parse_args()


def cross_entropy_loss(logits, targets, pad_id):
    N, S, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    logits_flat -= np.max(logits_flat, axis=1, keepdims=True)
    exp_logits = np.exp(logits_flat)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    mask = (targets_flat != pad_id)
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]

    loss = -np.log(correct_probs + 1e-9)
    loss = (loss * mask).sum() / mask.sum()

    dlogits = probs
    dlogits[np.arange(len(targets_flat)), targets_flat] -= 1
    dlogits = dlogits * mask[:, np.newaxis]
    dlogits /= mask.sum()

    return loss, dlogits.reshape(N, S, V)


def train():
    args = parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_encoder, args.vocab_merges)
    vocab_size = len(tokenizer.encoder)
    pad_id = tokenizer.encoder["<pad>"]

    pairs = load_parallel(args.input)
    print(f"Loaded {len(pairs)} parallel pairs from {len(args.input)} file(s)")

    if args.resume:
        print(f"Resuming from {args.resume}...")
        # Add module aliases for pickle compatibility with old model paths
        import model as model_pkg
        sys.modules['transformer'] = model_pkg.transformer
        sys.modules['encoder'] = model_pkg.encoder
        sys.modules['decoder'] = model_pkg.decoder
        sys.modules['encoder_layer'] = model_pkg.encoder_layer
        sys.modules['decoder_layer'] = model_pkg.decoder_layer
        sys.modules['mha'] = model_pkg.mha
        sys.modules['sdpa'] = model_pkg.sdpa
        sys.modules['embedding'] = model_pkg.embedding
        sys.modules['linear'] = model_pkg.linear
        sys.modules['layer_norm'] = model_pkg.layer_norm
        sys.modules['ff_network'] = model_pkg.ff_network
        sys.modules['dropout'] = model_pkg.dropout
        sys.modules['pe'] = model_pkg.pe
        with open(args.resume, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"Initializing Transformer (Vocab: {vocab_size}, d_model: {args.d_model})...")
        model = Transformer(vocab_size, args.d_model, args.num_heads, args.d_ff, args.num_layers, pad_id=pad_id)
    optimizer = Adam(model, lr=args.lr)

    print(f"Starting training for {args.steps} steps...")

    pbar = tqdm(range(args.steps), desc="Training")
    for step in pbar:
        src, decoder_input, targets = get_batch(pairs, tokenizer, args.batch_size, args.max_len)

        logits = model.forward(src, decoder_input, training=True)

        loss, grad_logits = cross_entropy_loss(logits, targets, pad_id)

        model.backward(grad_logits)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss:.4f}")

    print("Training Complete!")

    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    train()
