#!/usr/bin/env python3
"""
Search for exact phrases in training datasets (TinyStories, WikiData, etc.).

Useful for checking whether model-generated text is original or memorized
from training data.

Usage:
    python scripts/search_dataset.py --phrase "Once upon a time there was"
    python scripts/search_dataset.py --file generated_output.txt
    python scripts/search_dataset.py --phrase "..." --ngram 5
    python scripts/search_dataset.py --interactive
    python scripts/search_dataset.py --dataset tinystories --phrase "..."
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tinystories(split: str) -> list[tuple[int, str]]:
    """Load TinyStories texts. Returns list of (index, text) tuples."""
    from datasets import load_dataset

    if split == "both":
        splits = ["train", "validation"]
    else:
        splits = [split]

    results = []
    offset = 0
    for s in splits:
        ds = load_dataset("roneneldan/TinyStories", split=s)
        for i, item in enumerate(ds):
            text = item.get("text", "").strip()
            if text:
                results.append((offset + i, text))
        offset += len(ds)

    return results


def load_dataset_texts(dataset: str, split: str) -> list[tuple[int, str]]:
    """Load texts from the given dataset. Returns list of (index, text) tuples."""
    if dataset == "tinystories":
        return load_tinystories(split)
    # Add more datasets here, e.g.:
    # elif dataset == "wikidata":
    #     return load_wikidata(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: tinystories")


def search_phrase(
    texts: list[tuple[int, str]],
    phrase: str,
    context_chars: int = 80,
    max_matches: int = 10,
) -> list[dict]:
    """Search for exact substring matches. Returns list of match info dicts."""
    phrase = phrase.strip()
    if not phrase:
        return []

    matches = []
    for idx, text in texts:
        pos = text.find(phrase)
        while pos != -1:
            start = max(0, pos - context_chars)
            end = min(len(text), pos + len(phrase) + context_chars)
            context = text[start:end]
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."

            matches.append(
                {
                    "story_idx": idx,
                    "position": pos,
                    "length": len(phrase),
                    "context": context,
                }
            )
            if len(matches) >= max_matches:
                return matches
            pos = text.find(phrase, pos + 1)

    return matches


def ngram_overlap(
    texts: list[tuple[int, str]],
    query: str,
    n: int,
    min_length: int,
) -> dict:
    """Break query into N-word windows and report overlap with dataset."""
    words = query.split()
    if len(words) < n:
        return {"total_ngrams": 0, "found": 0, "fraction": 0.0, "found_ngrams": []}

    query_ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        if len(ngram) >= min_length:
            query_ngrams.append(ngram)

    found = []
    for ngram in query_ngrams:
        for _, text in texts:
            if ngram in text:
                found.append(ngram)
                break

    return {
        "total_ngrams": len(query_ngrams),
        "found": len(found),
        "fraction": len(found) / len(query_ngrams) if query_ngrams else 0.0,
        "found_ngrams": found[:20],  # Limit output
    }


def run_search(
    texts: list[tuple[int, str]],
    phrase: str,
    min_length: int,
    ngram: Optional[int],
    context_chars: int,
    max_matches: int,
) -> None:
    """Run search and print results."""
    phrase = phrase.strip()
    if not phrase:
        print("Empty phrase, skipping.")
        return

    if ngram is not None:
        result = ngram_overlap(texts, phrase, ngram, min_length)
        print(f"\nN-gram overlap (n={ngram}, min_length={min_length}):")
        print(f"  Total {ngram}-grams: {result['total_ngrams']}")
        print(f"  Found in dataset: {result['found']}")
        print(f"  Overlap fraction: {result['fraction']:.1%}")
        if result["found_ngrams"]:
            print("  Sample matches:")
            for ng in result["found_ngrams"][:10]:
                print(f"    - {repr(ng)}")
        return

    if len(phrase) < min_length:
        print(f"Phrase length ({len(phrase)}) < min_length ({min_length}), skipping exact search.")
        return

    matches = search_phrase(texts, phrase, context_chars, max_matches)
    total_stories = len(texts)

    print(f"\nSearch: {repr(phrase)}")
    print(f"Length: {len(phrase)} chars")
    print(f"Matches: {len(matches)} (showing up to {max_matches})")
    if matches:
        for m in matches:
            print(f"\n  Doc #{m['story_idx']} (pos {m['position']}, len {m['length']}):")
            print(f"  {m['context']}")
    else:
        print("  No exact matches found.")


def main():
    parser = argparse.ArgumentParser(
        description="Search for exact phrases in training datasets (TinyStories, WikiData, etc.)"
    )
    parser.add_argument(
        "--dataset",
        choices=["tinystories"],
        default="tinystories",
        help="Dataset to search (default: tinystories). More datasets coming (e.g. wikidata).",
    )
    parser.add_argument(
        "--phrase",
        type=str,
        help="Exact phrase to search for",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="File with generated output (each line/paragraph searched)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "both"],
        default="train",
        help="Dataset split to search (default: train)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="Min phrase length for exact search; min n-gram length for ngram mode (default: 20)",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=None,
        metavar="N",
        help="N-gram overlap mode: break query into N-word windows",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=80,
        help="Chars of context around match (default: 80)",
    )
    parser.add_argument(
        "--max_matches",
        type=int,
        default=10,
        help="Max exact matches to show (default: 10)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive REPL mode for multiple queries",
    )
    args = parser.parse_args()

    if not args.phrase and not args.file and not args.interactive:
        parser.error("Provide --phrase, --file, or --interactive")

    print(f"Loading {args.dataset} ({args.split})...")
    texts = load_dataset_texts(args.dataset, args.split)
    print(f"Loaded {len(texts)} documents.")

    def do_search(phrase: str) -> None:
        run_search(
            texts,
            phrase,
            min_length=args.min_length,
            ngram=args.ngram,
            context_chars=args.context,
            max_matches=args.max_matches,
        )

    if args.interactive:
        print("\nInteractive mode. Enter phrases to search (empty to quit):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            do_search(line)

    elif args.file:
        path = args.file
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        content = path.read_text()
        # Split by double newlines (paragraphs) or single newlines
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        if not blocks:
            blocks = [line.strip() for line in content.split("\n") if line.strip()]
        for i, block in enumerate(blocks):
            if len(block) >= args.min_length or args.ngram:
                print(f"\n--- Block {i + 1} ---")
                do_search(block)

    else:
        do_search(args.phrase)


if __name__ == "__main__":
    main()
