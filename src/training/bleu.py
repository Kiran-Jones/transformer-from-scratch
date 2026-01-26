# src/bleu.py
# Compute BLEU score for a seq2seq translation model on a test set

import argparse
import pickle
from pathlib import Path

import sacrebleu


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.tokenizer import Tokenizer
from inference.generator import Generator

from tqdm import tqdm



def iter_tsv_pairs(path: str):
    """Yield (src, ref) pairs from a TSV file (expects <en>\\t<es>)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            src, ref = line.split("\t", 1)
            src, ref = src.strip(), ref.strip()
            if src and ref:
                yield src, ref


def read_tsv_4col_slice(path: str, start_line: int, limit: int | None = None):
    """
    Read a slice from a 4-column TSV:
        <en_id>\t<english>\t<es_id>\t<spanish>

    Args:
        start_line: 1-based line number to start from
        limit: number of examples to read (None = all remaining)

    Returns:
        (srcs, refs) where srcs are English sentences, refs are Spanish references
    """
    srcs, refs = [], []
    current_line = 0  

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            current_line += 1
            if current_line < start_line:
                continue
            if limit is not None and len(srcs) >= limit:
                break

            line = raw.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                # Skip lines with formatting issues
                continue

            english = parts[1].strip()
            spanish = parts[3].strip()

            if english and spanish:
                srcs.append(english)
                refs.append(spanish)

    return srcs, refs, current_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Full TSV dataset")
    ap.add_argument("--train-lines", type=int, default=0,
                    help="Number of initial lines used for training (default: 0)")
    ap.add_argument("--test-size", type=int, default=None,
                    help="How many lines after train split to use for test. Default: all remaining.")
    ap.add_argument("--start-at", type=int, default=None,
                    help="1-based line index where test begins. Default: train_lines+1")

    ap.add_argument("-m", "--model", default="seq2seq_model.pkl",
                    help="Path to model weights file (default: seq2seq_model.pkl)")
    ap.add_argument("--max-len", type=int, default=200,
                    help="Max generation length (default: 200)")
    ap.add_argument("--vocab-encoder", default="src/vocab_encoder.json",
                    help="Path to vocab encoder JSON (default: src/vocab_encoder.json)")
    ap.add_argument("--vocab-merges", default="src/vocab_merges.json",
                    help="Path to vocab merges JSON (default: src/vocab_merges.json)")
    ap.add_argument("--beam-width", type=int, default=5,
                    help="Beam width for generation (default: 5)")

    ap.add_argument("--save-hyps", default=None,
                    help="Optional path to save hypotheses (one per line)")
    ap.add_argument("--save-test-tsv", default=None,
                    help="Optional path to save the exact test slice as TSV (<en>\\t<es>)")
    ap.add_argument("--lower", action="store_true",
                    help="Lowercase refs+hyps before BLEU (use if you trained lowercased)")
    ap.add_argument("--show", type=int, default=5,
                    help="Print N sample translations (default: 5)")

    args = ap.parse_args()

    test_start = args.start_at if args.start_at is not None else (args.train_lines + 1)
    if test_start < 1:
        raise ValueError("--start-at must be >= 1")

    # Load tokenizer + model + generator
    tokenizer = Tokenizer.from_files(args.vocab_encoder, args.vocab_merges)
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    generator = Generator(model, tokenizer)

    # Read test slice using the 4-column parser
    srcs, refs, last_line_seen = read_tsv_4col_slice(
        args.data,
        start_line=test_start,
        limit=args.test_size
    )

    if not srcs:
        raise RuntimeError(
            "No test examples found. Check --data path, split indices, and TSV column format."
        )

    # Optionally save the test slice in clean 2-col format
    if args.save_test_tsv:
        outp = Path(args.save_test_tsv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            for s, r in zip(srcs, refs):
                f.write(f"{s}\t{r}\n")
        print(f"Saved test slice TSV to: {outp}")

    # Generate translations
    hyps = []
    with tqdm(total=len(srcs), desc="Generating translations") as pbar:
        for s in srcs:
            h = generator.generate(s, max_len=args.max_len, beam_width=args.beam_width)
            hyps.append(h)
            pbar.update(1)  
 
    # Optional lowercasing
    if args.lower:
        hyps_eval = [h.lower() for h in hyps]
        refs_eval = [r.lower() for r in refs]
    else:
        hyps_eval = hyps
        refs_eval = refs

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(hyps_eval, [refs_eval])

    print("=" * 70)
    print(f"Data file: {args.data}")
    print(f"Train lines: 1..{args.train_lines}")
    if args.test_size is None:
        # We don't know exact end-of-file without another pass; show last line reached in this read
        print(f"Test lines: {test_start}..(end)  (examples used: {len(srcs)})")
    else:
        end_line = test_start + args.test_size - 1
        print(f"Test lines: {test_start}..{end_line}  (examples used: {len(srcs)})")
    print(f"BLEU (SacreBLEU): {bleu.score:.2f}")
    print(f"Precisions (1-4g): {bleu.precisions}")
    print(f"BP: {bleu.bp:.4f} | sys_len/ref_len: {bleu.sys_len}/{bleu.ref_len}")
    print("=" * 70)

    # Save hypotheses if requested
    if args.save_hyps:
        outp = Path(args.save_hyps)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            for h in hyps:
                f.write(h.strip() + "\n")
        print(f"Saved hypotheses: {outp}")

    # Show samples
    n = min(args.show, len(srcs))
    if n > 0:
        print("\nSamples:")
        print("-" * 70)
        for i in range(n):
            print(f"EN:  {srcs[i]}")
            print(f"REF: {refs[i]}")
            print(f"HYP: {hyps[i]}")
            print("-" * 70)


if __name__ == "__main__":
    main()
