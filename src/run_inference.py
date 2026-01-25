import argparse
import pickle
from tokenizer import Tokenizer
from generator import Generator


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Transformer")
    parser.add_argument("-m", "--model", default="model_weights.pkl",
                        help="Path to model weights file (default: model_weights.pkl)")
    parser.add_argument("-p", "--prompt", nargs="+", required=True,
                        help="Prompt(s) to generate from")
    parser.add_argument("--max-len", type=int, default=100,
                        help="Max generation length (default: 100)")
    parser.add_argument("--beam-width", type=int, default=4,
                        help="Beam search width (default: 4, use 1 for greedy)")
    parser.add_argument("--vocab-encoder", default="src/vocab_encoder.json",
                        help="Path to vocab encoder JSON (default: src/vocab_encoder.json)")
    parser.add_argument("--vocab-merges", default="src/vocab_merges.json",
                        help="Path to vocab merges JSON (default: src/vocab_merges.json)")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_encoder, args.vocab_merges)

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    generator = Generator(model, tokenizer)

    print("-" * 40)
    for prompt in args.prompt:
        output = generator.generate(prompt, max_len=args.max_len, beam_width=args.beam_width)
        print(f"Input:  {prompt}")
        print(f"Output: {output}")
        print("-" * 40)


if __name__ == "__main__":
    main()
