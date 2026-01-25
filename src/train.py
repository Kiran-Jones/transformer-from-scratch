import argparse
import numpy as np
from tqdm import tqdm
from transformer import Transformer
from tokenizer import Tokenizer
from optimizer import Adam


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="Input text file(s) for training (one sentence per line)")
    parser.add_argument("--vocab-encoder", default="src/vocab_encoder.json",
                        help="Path to vocab encoder JSON (default: src/vocab_encoder.json)")
    parser.add_argument("--vocab-merges", default="src/vocab_merges.json",
                        help="Path to vocab merges JSON (default: src/vocab_merges.json)")
    parser.add_argument("-o", "--output", default="model_weights.pkl",
                        help="Output model weights path (default: model_weights.pkl)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of training steps (default: 200)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--max-len", type=int, default=20,
                        help="Max sequence length (default: 20)")
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


def load_corpus(filepaths):
    """Load training sentences from text files (one sentence per line)."""
    corpus = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus.append(line)
    return corpus

def get_batch(data_corpus, tokenizer, batch_size, max_len):
    """Split each sentence into a source prefix and target suffix."""
    pad_id = tokenizer.encoder['<pad>']
    sos_id = tokenizer.encoder['<SOS>']
    eos_id = tokenizer.encoder['<EOS>']

    indices = np.random.randint(0, len(data_corpus), batch_size)
    batch_data = [data_corpus[i] for i in indices]

    src_batch = []
    tgt_batch = []

    for sentence in batch_data:
        tokens = tokenizer.encode(sentence)
        if len(tokens) < 2:
            tokens = tokens + tokens  # ensure at least 2 tokens

        # Split at a random point (at least 1 token on each side)
        split = np.random.randint(1, max(2, len(tokens)))
        src_tokens = tokens[:split]
        tgt_tokens = tokens[split:]

        # Encoder input: prefix (no SOS/EOS needed for encoder)
        src = src_tokens[:max_len]
        src = src + [pad_id] * (max_len - len(src))

        # Decoder input: <SOS> + suffix (teacher forcing)
        # Target: suffix + <EOS>
        dec_in = [sos_id] + tgt_tokens
        dec_tgt = tgt_tokens + [eos_id]

        dec_in = dec_in[:max_len]
        dec_tgt = dec_tgt[:max_len]

        dec_in = dec_in + [pad_id] * (max_len - len(dec_in))
        dec_tgt = dec_tgt + [pad_id] * (max_len - len(dec_tgt))

        src_batch.append(src)
        tgt_batch.append((dec_in, dec_tgt))

    src_arr = np.array(src_batch)
    dec_in_arr = np.array([t[0] for t in tgt_batch])
    dec_tgt_arr = np.array([t[1] for t in tgt_batch])

    return src_arr, dec_in_arr, dec_tgt_arr


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
    import pickle

    args = parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_encoder, args.vocab_merges)
    vocab_size = len(tokenizer.encoder)
    data_corpus = load_corpus(args.input)
    print(f"Loaded {len(data_corpus)} sentences from {len(args.input)} file(s)")

    pad_id = tokenizer.encoder["<pad>"]
    if args.resume:
        print(f"Resuming from {args.resume}...")
        with open(args.resume, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"Initializing Transformer (Vocab: {vocab_size})...")
        model = Transformer(vocab_size, args.d_model, args.num_heads, args.d_ff, args.num_layers, pad_id=pad_id)
    optimizer = Adam(model, lr=args.lr)

    print(f"Starting training for {args.steps} steps...")

    pbar = tqdm(range(args.steps), desc="Training")
    for step in pbar:
        src, decoder_input, targets = get_batch(data_corpus, tokenizer, args.batch_size, args.max_len)

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
