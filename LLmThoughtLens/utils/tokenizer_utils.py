"""Tokenizer helpers used when no model-side tokenizer is exposed."""

from __future__ import annotations

from collections.abc import Iterable

MASK_TOKEN = "[MASK]"


def whitespace_tokens(text: str) -> list[str]:
    """Whitespace tokeniser. Returns ``['<empty>']`` for empty input."""
    parts = text.split()
    return parts if parts else ["<empty>"]


def replace_token(tokens: list[str], idx: int, replacement: str = MASK_TOKEN) -> list[str]:
    """Return a copy of *tokens* with the token at *idx* replaced."""
    if not 0 <= idx < len(tokens):
        raise IndexError(f"token index {idx} out of bounds for length {len(tokens)}")
    out = list(tokens)
    out[idx] = replacement
    return out


def mask_positions(
    tokens: list[str], indices: Iterable[int], replacement: str = MASK_TOKEN
) -> list[str]:
    """Return a copy of *tokens* with every position in *indices* replaced."""
    out = list(tokens)
    n = len(out)
    for i in indices:
        if 0 <= i < n:
            out[i] = replacement
    return out


def token_join(tokens: list[str]) -> str:
    """Join tokens back into a single string (whitespace-separated)."""
    return " ".join(tokens)
