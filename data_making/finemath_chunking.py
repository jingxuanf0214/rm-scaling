from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from typing import Optional
from tqdm import tqdm
import os
import argparse

def prepare_cpt_blocks(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    text_column: str = "text",
    tokenizer_id: str = "meta-llama/Meta-Llama-3-8B",  # use your Llama tokenizer here
    block_size: int = 4096,
    prefix_len: int = 2048,
    overlap_size: int = 0,
    shuffle_seed: Optional[int] = 42,
    save_dir: Optional[str] = None,
    filter_url: bool = False,
):
    """
    Build fixed-size token blocks from a HF dataset of text docs, using a Llama tokenizer.
    Each block is split into 'prefix' and 'target'.

    Args:
        overlap_size: Number of tokens to overlap between consecutive chunks.
                      If 0 (default), chunks are non-overlapping.
                      E.g., with block_size=1536 and overlap_size=300:
                        chunk 1: tokens 0-1535
                        chunk 2: tokens 1236-2771 (stride = 1536-300 = 1236)

    Returns:
        hf_out (datasets.Dataset): with columns 'prefix' and 'target'.
    Optionally saves to disk with .save_to_disk(save_dir).
    """
    assert block_size > 0 and prefix_len > 0 and prefix_len < block_size, \
        "prefix_len must be in (0, block_size)"
    assert block_size % 1 == 0 and prefix_len % 1 == 0
    assert 0 <= overlap_size < block_size, \
        f"overlap_size must be in [0, block_size), got {overlap_size}"
    target_len = block_size - prefix_len
    assert target_len > 0, "target must be non-empty"
    
    # Stride is how much we advance after each block
    stride = block_size - overlap_size
    assert stride > 0, "stride must be positive (overlap_size must be < block_size)"

    # 1) Load dataset
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    # Optional: filter by URL to only keep StackExchange content
    if filter_url:
        print("Filtering dataset by URL to stackexchange content only...")
        if 'url' in ds.column_names:
            before_n = len(ds)
            ds = ds.filter(lambda x: 'stackexchange' in x['url'])
            print(f"Total samples after URL filtering: {len(ds)} (from {before_n})")
        else:
            print("Warning: 'url' column not found; skipping URL filtering.")

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)

    # 2) Load tokenizer (Llama family). Use the exact tokenizer matching your base model.
    tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; set one or choose a Llama tokenizer.")
    eos_id = tok.eos_token_id

    # 3) Tokenize without adding special tokens automatically (we'll append EOS exactly once per doc)
    def _tok(batch):
        out = tok(batch[text_column], add_special_tokens=False)
        return {"_input_ids": out["input_ids"], "_n": [len(x) for x in out["input_ids"]]}

    ds_tok = ds.map(_tok, batched=True, remove_columns=ds.column_names)

    # 4) Concatenate + pack to fixed blocks via a generator (efficient, minimal RAM)
    def block_generator():
        buf = []
        for row in tqdm(ds_tok, desc=f"Packing {block_size}-token blocks (stride={stride})"):
            ids = row["_input_ids"]
            # append exactly one EOS per document
            if not ids or ids[-1] != eos_id:
                ids = ids + [eos_id]
            buf.extend(ids)

            # flush full blocks with overlap (stride = block_size - overlap_size)
            while len(buf) >= block_size:
                block = buf[:block_size]
                del buf[:stride]  # advance by stride, keeping overlap for next block
                # split into prefix/target
                yield {
                    "prefix": block[:prefix_len],
                    "target": block[prefix_len:],
                }
        # drop the final remainder (< block_size) to avoid padding

    features = Features({
        "prefix": Sequence(Value("int32")),
        "target": Sequence(Value("int32")),
    })

    hf_out = Dataset.from_generator(block_generator, features=features)

    if save_dir:
        save_path = os.path.join(save_dir, subset if subset else "default")
        os.makedirs(save_path, exist_ok=True)
        hf_out.save_to_disk(save_path)

    return hf_out


def main():
    parser = argparse.ArgumentParser(description="Prepare CPT blocks from a HuggingFace dataset")
    
    # Optional arguments with defaults
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/finemath", help="Name of the HuggingFace dataset to load")
    parser.add_argument("--subset", type=str, default="finemath-4plus", help="Dataset subset/configuration name (e.g., 'en' for c4, '20220301.en' for wikipedia)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column (default: text)")
    parser.add_argument("--tokenizer_id", type=str, default="meta-llama/Meta-Llama-3-8B", 
                       help="Tokenizer ID to use (default: meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--block_size", type=int, default=4096, help="Block size in tokens (default: 4096)")
    parser.add_argument("--prefix_len", type=int, default=2048, help="Prefix length in tokens (default: 2048)")
    parser.add_argument("--overlap_size", type=int, default=0, 
                       help="Number of tokens to overlap between consecutive chunks (default: 0, no overlap)")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling")
    parser.add_argument("--save_dir", type=str, default="../data/finemath_highquality_cpt_4plus", help="Directory to save the processed dataset")
    parser.add_argument("--filter_url", action="store_true", help="If set, filter dataset to rows where 'url' contains 'stackexchange'")
    
    args = parser.parse_args()
    
    # Handle shuffle seed
    shuffle_seed = None if args.no_shuffle else args.shuffle_seed
    
    print(f"Processing dataset: {args.dataset_name}")
    if args.subset:
        print(f"Subset: {args.subset}")
    print(f"Split: {args.split}")
    print(f"Text column: {args.text_column}")
    print(f"Tokenizer: {args.tokenizer_id}")
    print(f"Block size: {args.block_size}")
    print(f"Prefix length: {args.prefix_len}")
    print(f"Overlap size: {args.overlap_size}")
    print(f"Stride: {args.block_size - args.overlap_size}")
    print(f"Shuffle seed: {shuffle_seed}")
    print(f"Save directory: {args.save_dir}")
    print(f"Filter URL (stackexchange): {args.filter_url}")
    
    # Call the function
    dataset = prepare_cpt_blocks(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        text_column=args.text_column,
        tokenizer_id=args.tokenizer_id,
        block_size=args.block_size,
        prefix_len=args.prefix_len,
        overlap_size=args.overlap_size,
        shuffle_seed=shuffle_seed,
        save_dir=args.save_dir,
        filter_url=args.filter_url,
    )
    
    print(f"\nProcessing complete!")
    print(f"Generated {len(dataset)} blocks")
    if args.save_dir:
        print(f"Dataset saved to: {args.save_dir}")
    else:
        print("Dataset not saved (no --save_dir specified)")


if __name__ == "__main__":
    main()
