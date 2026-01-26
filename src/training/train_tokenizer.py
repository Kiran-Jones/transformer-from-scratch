import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import regex as re
from collections import Counter
from inference.tokenizer import Tokenizer


def get_stats(vocab):
    """Compute frequencies of adjacent pairs in the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """Replace all occurrences of 'pair' in the vocab with a new merged token."""
    v_out = {}
    raw_bigram = ' '.join(pair)
    escaped_bigram = re.escape(pair[0]) + r' ' + re.escape(pair[1])
    p = re.compile(r'(?<!\S)' + escaped_bigram + r'(?!\S)')
    for word in v_in:
        if raw_bigram not in word:
            v_out[word] = v_in[word]
            continue
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def train_tokenizer(text, vocab_size=500):
    """Train BPE tokenizer on raw text."""
    print(f"Training tokenizer on {len(text)} characters...")

    temp_tokenizer = Tokenizer()

    print("Pre-tokenizing text...")
    words = re.findall(temp_tokenizer.pattern, text)
    print(f"Found {len(words)} words.")

    vocab = Counter()
    for word in words:
        token_bytes = word.encode('utf-8')
        token_translated = [temp_tokenizer.byte_encoder[b] for b in token_bytes]
        vocab[' '.join(token_translated)] += 1

    merges = {}
    num_merges = vocab_size - 256

    print(f"Starting BPE merge (target {vocab_size} tokens)...")
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges[best] = i

        if (i + 1) % 10 == 0:
            print(f"Merge {i+1}/{num_merges}: {best} (freq: {pairs[best]})")

    encoder = {v: k for k, v in temp_tokenizer.byte_encoder.items()}

    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    for (p1, p2), rank in sorted_merges:
        token = p1 + p2
        encoder[token] = len(encoder)

    print(f"Training complete. Final Vocab size: {len(encoder)}")
    return encoder, merges


def save_tokenizer(encoder, merges, filename_prefix="src/vocab"):
    os.makedirs(os.path.dirname(filename_prefix) or '.', exist_ok=True)

    with open(f"{filename_prefix}_encoder.json", "w", encoding="utf-8") as f:
        json.dump(encoder, f, ensure_ascii=False, indent=2)

    merges_list = list(merges.keys())
    with open(f"{filename_prefix}_merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_list, f, ensure_ascii=False, indent=2)
    print(f"Saved tokenizer to {filename_prefix}_encoder.json and _merges.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("-i", "--input", nargs="+",
                        help="Input text file(s) to train on")
    parser.add_argument("--tsv", type=str,
                        help="Input TSV file to extract text from")
    parser.add_argument("--columns", nargs="+", type=int, default=[1, 3],
                        help="0-indexed column indices to extract from TSV (default: 1 3)")
    parser.add_argument("--vocab-size", type=int, default=300,
                        help="Target vocabulary size (default: 300)")
    parser.add_argument("--output-prefix", default="src/vocab",
                        help="Output file prefix (default: src/vocab)")
    args = parser.parse_args()

    if not args.input and not args.tsv:
        parser.error("Either --input or --tsv is required")

    text = ""
    if args.input:
        for filepath in args.input:
            with open(filepath, "r", encoding="utf-8") as f:
                text += f.read() + "\n"

    if args.tsv:
        with open(args.tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                for col in args.columns:
                    if col < len(parts) and parts[col].strip():
                        text += parts[col].strip() + "\n"

    encoder, merges = train_tokenizer(text, vocab_size=args.vocab_size)
    save_tokenizer(encoder, merges, filename_prefix=args.output_prefix)
