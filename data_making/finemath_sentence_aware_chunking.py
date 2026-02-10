from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from typing import Optional, List
from tqdm import tqdm
import os
import argparse
import re

# -------------------------
# Sentence splitting helpers
# -------------------------

# Small abbreviation list to reduce false splits on "e.g.", "Fig.", etc.
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "cf", "fig", "eq", "ref", "no", "dr", "mr", "ms", "mrs", "prof",
    "al", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# Characters that often appear immediately after sentence-ending punctuation.
_TRAILING_QUOTES = "\"')]}"

# Characters that a new sentence might start with.
_LEADING_SENT_CHARS = "\"'([{"

def split_into_sentences(text: str) -> List[str]:
    """Heuristic sentence splitter (no external deps).

    - Treat blank lines as hard boundaries (paragraphs).
    - Split on .!? when next non-space char looks like a sentence start.
    - Avoid splitting on common abbreviations.
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out: List[str] = []

    def is_abbrev(prev_word: str) -> bool:
        w = prev_word.strip().strip(_TRAILING_QUOTES + "([{").rstrip(".").lower()
        return w in _ABBREVIATIONS

    for para in paragraphs:
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue

        start = 0
        i = 0
        n = len(para)

        while i < n:
            ch = para[i]
            if ch in ".!?":
                # Optional closing quotes/brackets immediately after punctuation.
                j = i + 1
                while j < n and para[j] in _TRAILING_QUOTES:
                    j += 1

                # Require whitespace after punctuation/quotes.
                if j < n and para[j].isspace():
                    prev_chunk = para[start:i].rstrip()
                    prev_word = prev_chunk.split()[-1] if prev_chunk.split() else ""
                    if not is_abbrev(prev_word):
                        # Find next non-space char.
                        k = j
                        while k < n and para[k].isspace():
                            k += 1
                        if k >= n:
                            sent = para[start:].strip()
                            if sent:
                                out.append(sent)
                            start = n
                            break

                        nxt = para[k]
                        looks_like_start = (
                            nxt.isupper()
                            or nxt.isdigit()
                            or nxt in _LEADING_SENT_CHARS
                        )
                        if looks_like_start:
                            sent = para[start:j].strip()
                            if sent:
                                out.append(sent)
                            start = k
                            i = k
                            continue
            i += 1

        if start < n:
            rem = para[start:].strip()
            if rem:
                out.append(rem)

    return out


def prepare_cpt_blocks(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    text_column: str = "text",
    tokenizer_id: str = "meta-llama/Meta-Llama-3-8B",
    block_size: int = 4096,
    prefix_len: int = 2048,
    overlap_size: int = 0,
    shuffle_seed: Optional[int] = 42,
    save_dir: Optional[str] = None,
    filter_url: bool = False,
    sentence_aware: bool = True,
    block_tolerance: int = 128,
    long_sentence_strategy: str = "split",  # split|drop
):
    """Build CPT prefix/target pairs; optionally keep boundaries on sentences.

    sentence_aware=True:
      - Split each document into sentences.
      - Pack blocks by whole sentences only (never cut a sentence),
        allowing blocks to be slightly shorter than block_size.
      - Choose prefix/target split on a sentence boundary closest to prefix_len.
      - Approximate overlap_size using whole sentences.

    If a single "sentence" tokenizes longer than block_size, sentence-preserving is impossible:
      - long_sentence_strategy="split": hard-split that sentence as a last resort.
      - long_sentence_strategy="drop": skip it.
    """
    assert 0 < prefix_len < block_size
    assert 0 <= overlap_size < block_size
    assert 0 <= block_tolerance < block_size
    assert long_sentence_strategy in {"split", "drop"}

    # 1) Load dataset
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if filter_url:
        print("Filtering dataset by URL to stackexchange content only...")
        if "url" in ds.column_names:
            before_n = len(ds)
            ds = ds.filter(lambda x: "stackexchange" in x["url"])
            print(f"Total samples after URL filtering: {len(ds)} (from {before_n})")
        else:
            print("Warning: 'url' column not found; skipping URL filtering.")

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; set one or choose a Llama tokenizer.")
    eos_id = tok.eos_token_id

    min_block_len = max(2, block_size - block_tolerance)

    def flatten(chunks: List[List[int]]) -> List[int]:
        out: List[int] = []
        for c in chunks:
            out.extend(c)
        return out

    def choose_boundary(sent_tok: List[List[int]]) -> Optional[int]:
        """Pick boundary i (1..n-1) so prefix is closest to prefix_len."""
        if len(sent_tok) < 2:
            return None
        cum = 0
        best_i = None
        best_diff = None
        for i in range(1, len(sent_tok)):
            cum += len(sent_tok[i - 1])
            diff = abs(cum - prefix_len)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_i = i
        return best_i

    def make_overlap(sent_lists: List[List[int]]) -> List[List[int]]:
        if overlap_size <= 0:
            return []
        keep: List[List[int]] = []
        total = 0
        for s in reversed(sent_lists):
            if total >= overlap_size:
                break
            keep.append(s)
            total += len(s)
        keep.reverse()
        return keep

    def block_generator():
        cur_sents: List[List[int]] = []
        cur_len = 0

        def emit_if_ready():
            nonlocal cur_sents, cur_len
            if cur_len < min_block_len:
                return None

            split_i = choose_boundary(cur_sents)
            if split_i is None:
                return None

            prefix = flatten(cur_sents[:split_i])
            target = flatten(cur_sents[split_i:])
            if not prefix or not target:
                return None

            # Reset block to overlap
            ex = {"prefix": prefix, "target": target}
            cur_sents = make_overlap(cur_sents)
            cur_len = sum(len(s) for s in cur_sents)
            return ex

        for row in tqdm(ds, desc=f"Packing blocks (sentence_aware={sentence_aware})"):
            text = row.get(text_column, "")
            if not isinstance(text, str) or not text:
                continue

            if sentence_aware:
                sents = split_into_sentences(text)
                if not sents:
                    continue
                sent_ids = tok(sents, add_special_tokens=False)["input_ids"]
                sent_ids.append([eos_id])  # EOS as its own boundary token
            else:
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if not ids or ids[-1] != eos_id:
                    ids = ids + [eos_id]
                sent_ids = [ids]  # one big chunk

            for sent in sent_ids:
                if not sent:
                    continue

                # Sentence longer than block_size: can't keep it intact.
                if len(sent) > block_size:
                    if long_sentence_strategy == "drop":
                        continue
                    # Hard-split this one "sentence" as a last resort.
                    for i in range(0, len(sent), block_size):
                        chunk = sent[i:i + block_size]

                        ex = emit_if_ready()
                        if ex is not None:
                            yield ex

                        cur_sents = [chunk]
                        cur_len = len(chunk)
                        ex = emit_if_ready()
                        if ex is not None:
                            yield ex
                    continue

                # Normal packing: add whole sentence if it fits; else flush.
                if cur_len + len(sent) <= block_size:
                    cur_sents.append(sent)
                    cur_len += len(sent)
                else:
                    ex = emit_if_ready()
                    if ex is not None:
                        yield ex
                    else:
                        # Too short/no boundary to emit; reset (keep overlap if requested).
                        cur_sents = make_overlap(cur_sents) if overlap_size > 0 else []
                        cur_len = sum(len(s) for s in cur_sents)

                    # Add sentence; if overlap makes it not fit, drop overlap.
                    if cur_len + len(sent) <= block_size:
                        cur_sents.append(sent)
                        cur_len += len(sent)
                    else:
                        cur_sents = [sent]
                        cur_len = len(sent)

        # Drop final remainder (< min_block_len). If you want it, lower block_tolerance
        # and/or pad in your training collator.

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
    parser = argparse.ArgumentParser(description="Prepare CPT blocks from a HF dataset using a Llama tokenizer.")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/finemath")
    parser.add_argument("--subset", type=str, default="finemath-4plus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--tokenizer_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--prefix_len", type=int, default=2048)
    parser.add_argument("--overlap_size", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--filter_url", action="store_true")

    # New knobs
    parser.add_argument("--sentence_aware", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--block_tolerance", type=int, default=128)
    parser.add_argument("--long_sentence_strategy", type=str, default="split", choices=["split", "drop"])

    args = parser.parse_args()
    shuffle_seed = None if args.shuffle_seed == -1 else args.shuffle_seed

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
        sentence_aware=args.sentence_aware,
        block_tolerance=args.block_tolerance,
        long_sentence_strategy=args.long_sentence_strategy,
    )

    print(f"Generated {len(dataset)} blocks")
    if args.save_dir:
        print(f"Saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
