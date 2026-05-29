"""Tokenizer helpers used when a real tokenizer isn't available.

The mock and the black-box providers fall back to whitespace tokenisation;
this module centralises that logic and a small reusable mask helper.
"""

from __future__ import annotations

from collections.abc import Iterable

MASK_TOKEN = "[MASK]"


def whitespace_tokens(text: str) -> list[str]:
    """Whitespace tokeniser preserving non-empty fragments.

    Used by black-box providers when no model-side tokeniser is exposed.
    Returns ``["<empty>"]`` for the empty string so downstream code never
    has to handle a zero-length token sequence.
    """
    parts = text.split()
    return parts if parts else ["<empty>"]


def replace_token(tokens: list[str], idx: int, replacement: str = MASK_TOKEN) -> list[str]:
    """Return a new list with the token at *idx* replaced by *replacement*."""
    if not 0 <= idx < len(tokens):
        raise IndexError(f"token index {idx} out of bounds for length {len(tokens)}")
    out = list(tokens)
    out[idx] = replacement
    return out


def mask_positions(tokens: list[str], indices: Iterable[int], replacement: str = MASK_TOKEN) -> list[str]:
    """Replace every position in *indices* with *replacement*."""
    out = list(tokens)
    n = len(out)
    for i in indices:
        if 0 <= i < n:
            out[i] = replacement
    return out


def token_join(tokens: list[str]) -> str:
    """Join tokens back into a single string (whitespace-separated)."""
    return " ".join(tokens)
