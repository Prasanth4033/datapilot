"""How text becomes tokens, and how a chat call's tokens are counted.

Uses OpenAI's real tokenizer (tiktoken) when it is installed and can load its vocabulary.
Otherwise falls back to an approximate split, so everything still runs offline.
"""
from __future__ import annotations

import re
from functools import lru_cache

# OpenAI chat format adds a few hidden tokens around every message.
TOKENS_PER_MESSAGE = 3   # role marker + start/end markers for each message
REPLY_PRIMING = 3        # tokens that start the assistant's reply


@lru_cache(maxsize=None)
def _encoder(model: str):
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")   # encoding used by GPT-4o models
    except Exception:
        return None   # tiktoken not installed, or vocabulary file could not be downloaded


def tokenizer_name(model: str) -> str:
    enc = _encoder(model)
    return f"tiktoken ({enc.name})" if enc else "approximate (install tiktoken for exact counts)"


def tokenize(text: str, model: str = "gpt-4o-mini") -> list[tuple[int | None, str]]:
    """Return (token_id, piece) pairs. With the fallback, ids are None."""
    enc = _encoder(model)
    if enc:
        return [(i, enc.decode_single_token_bytes(i).decode("utf-8", "replace")) for i in enc.encode(text)]
    # Fallback: words keep their leading space; long words split into ~4-character pieces.
    pieces = []
    for word in re.findall(r" ?[A-Za-z]+| ?\d{1,3}| ?[^\sA-Za-z\d]|\s+", text):
        while len(word) > 5:
            pieces.append(word[:4])
            word = word[4:]
        pieces.append(word)
    return [(None, p) for p in pieces]


def count(text: str, model: str = "gpt-4o-mini") -> int:
    return len(tokenize(text, model))


def chat_breakdown(system: str, user: str, model: str = "gpt-4o-mini") -> dict:
    """Tokens the provider will bill as input for one chat call."""
    s, u = count(system, model), count(user, model)
    overhead = 2 * TOKENS_PER_MESSAGE + REPLY_PRIMING
    return {"system": s, "question": u, "overhead": overhead, "total": s + u + overhead}
