#!/usr/bin/env python3
"""Pre-tokenize training data and save as Arrow dataset.
Run inside the training container or on the host with the right env.

Usage:
    python pretokenize.py <jsonl_path> <output_dir> [--model-path <path>]
"""
import argparse, json, os, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory for tokenized dataset")
    parser.add_argument("--model-path", default="/workspace/models/merged",
                        help="Path to model for tokenizer")
    parser.add_argument("--max-len", type=int, default=24576)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/code/data")
    import schema
    from transformers import AutoTokenizer
    from datasets import Dataset, Features, Sequence, Value

    print(f"Loading tokenizer from {args.model_path}...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    def _gen(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ex = json.loads(ln)
                except Exception:
                    continue
                enc = schema.encode_example(
                    {"messages": ex["messages"], "tools": ex.get("tools")},
                    tok, max_len=args.max_len
                )
                if enc:
                    yield {
                        "input_ids": enc["input_ids"],
                        "labels": enc["labels"],
                        "attention_mask": enc["attention_mask"],
                    }

    feats = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })

    print(f"Tokenizing {args.jsonl_path}...")
    ds = Dataset.from_generator(
        _gen,
        gen_kwargs={"path": args.jsonl_path},
        features=feats,
    )
    print(f"Tokenized {len(ds)} examples")

    print(f"Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
