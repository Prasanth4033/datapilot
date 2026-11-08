"""Token explainer: shows, step by step, how a word becomes tokens.

It trains a tiny BPE tokenizer (the same method GPT models use) on a small text,
then splits your word with the learned merges and counts the tokens.

Usage
  python -m scripts.token_explainer lower lowest slow
  python -m scripts.token_explainer payment --merges 20
  python -m scripts.token_explainer DPD --corpus my_text.txt
  python -m scripts.token_explainer                    (type words interactively)
Options
  --merges N        how many merges to learn (default 12). More merges = bigger vocabulary = fewer tokens
  --corpus FILE     learn from your own text file instead of the built-in one
  --hide-training   do not print the training rounds
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

BUILT_IN_CORPUS = """
low low low low lower lower lowest slow slow
loan loan loan loans loans payment payment payments
late late late fee fee fees balance balance
newer newest wider widest
"""


def words_of(text: str) -> Counter:
    return Counter(re.findall(r"[A-Za-z]+", text.lower()))


def pair_counts(split_words: dict[str, list[str]], freq: Counter) -> Counter:
    counts = Counter()
    for word, pieces in split_words.items():
        for a, b in zip(pieces, pieces[1:]):
            counts[(a, b)] += freq[word]
    return counts


def merge_pieces(pieces: list[str], pair: tuple[str, str]) -> list[str]:
    out, i = [], 0
    while i < len(pieces):
        if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == pair:
            out.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


def train(corpus: str, num_merges: int, show: bool = True) -> list[tuple[str, str]]:
    """Learn merges: repeatedly glue the most frequent neighbouring pair."""
    freq = words_of(corpus)
    split_words = {w: list(w) for w in freq}
    merges: list[tuple[str, str]] = []
    if show:
        print("=" * 64)
        print("STEP 1 - TRAINING: learn which letters often sit together")
        print("=" * 64)
        print("Training words (how often each appears):")
        print("  " + ", ".join(f"{w} x{c}" for w, c in freq.most_common()))
        print("Every word starts as single letters, e.g. 'lower' -> l o w e r\n")
    for rnd in range(1, num_merges + 1):
        counts = pair_counts(split_words, freq)
        if not counts:
            break
        pair, seen = counts.most_common(1)[0]
        if seen < 2:
            if show:
                print(f"Round {rnd}: no pair appears 2+ times. Stop.")
            break
        merges.append(pair)
        split_words = {w: merge_pieces(p, pair) for w, p in split_words.items()}
        if show:
            print(f"Round {rnd:2}: most frequent pair '{pair[0]}' + '{pair[1]}' (seen {seen} times) "
                  f"-> new piece '{pair[0] + pair[1]}'")
    if show:
        print()
    return merges


def build_vocab(merges: list[tuple[str, str]]) -> dict[str, int]:
    """Give every piece an ID: letters a-z get 0-25, each learned piece the next number."""
    vocab = {chr(ord("a") + i): i for i in range(26)}
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    return vocab


def encode(word: str, merges: list[tuple[str, str]], show: bool = True) -> list[str]:
    """Split a word: start from letters, apply learned merges in the order they were learned."""
    pieces = list(word.lower())
    rank = {pair: i for i, pair in enumerate(merges)}
    if show:
        print(f"  Start (one piece per letter): {' | '.join(pieces)}   -> {len(pieces)} pieces")
    step = 1
    while True:
        candidates = [(rank[p], p) for p in zip(pieces, pieces[1:]) if p in rank]
        if not candidates:
            break
        r, pair = min(candidates)          # the merge learned earliest wins
        pieces = merge_pieces(pieces, pair)
        if show:
            print(f"  Step {step}: glue '{pair[0]}' + '{pair[1]}' (learned in round {r + 1})"
                  f"  -> {' | '.join(pieces)}   -> {len(pieces)} pieces")
        step += 1
    if show and step == 1:
        print("  No learned merge fits this word, so every letter stays a separate token.")
    return pieces


def explain(word: str, merges, vocab) -> None:
    print("-" * 64)
    print(f"WORD: '{word}'")
    print("-" * 64)
    pieces = encode(word, merges)
    ids = [vocab.get(p, -1) for p in pieces]
    print(f"\n  RESULT: '{word}' = {len(pieces)} token(s)")
    print("  Tokens: " + "  ".join(f"[{p}]" for p in pieces))
    print("  IDs:    " + "  ".join(str(i) for i in ids) + "   (the model only sees these numbers)")
    real = real_tokens(word)
    if real:
        print(f"\n  For comparison, GPT-4o's real tokenizer: {len(real)} token(s): "
              + "  ".join(f"[{p}]" for p in real))
    print()


def real_tokens(word: str) -> list[str] | None:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return [enc.decode([i]) for i in enc.encode(word)]
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("words", nargs="*")
    p.add_argument("--merges", type=int, default=12)
    p.add_argument("--corpus")
    p.add_argument("--hide-training", action="store_true")
    a = p.parse_args()

    corpus = Path(a.corpus).read_text(encoding="utf-8") if a.corpus else BUILT_IN_CORPUS
    merges = train(corpus, a.merges, show=not a.hide_training)
    vocab = build_vocab(merges)
    print("=" * 64)
    print("STEP 2 - SPLITTING: break your word into the learned pieces")
    print("=" * 64)
    print(f"Learned pieces: {', '.join(a + b for a, b in merges)}\n")

    if a.words:
        for w in a.words:
            explain(w, merges, vocab)
        return
    while True:
        try:
            w = input("Type a word (or exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not w or w.lower() == "exit":
            break
        explain(w, merges, vocab)


if __name__ == "__main__":
    main()
