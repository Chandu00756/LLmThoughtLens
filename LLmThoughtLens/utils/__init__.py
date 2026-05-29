"""Utility helpers used across LLmThoughtLens — colors, math, tokenizer."""

from LLmThoughtLens.utils.colors import (
    THOUGHTLENS_COLORS,
    edge_color,
    heatmap_colorscale,
    node_color,
)
from LLmThoughtLens.utils.math_utils import (
    cosine_sim,
    l2_normalise,
    pca_2d,
    softmax,
    topk_indices,
    topk_mask,
)
from LLmThoughtLens.utils.tokenizer_utils import (
    MASK_TOKEN,
    mask_positions,
    replace_token,
    token_join,
    whitespace_tokens,
)

__all__ = [
    "THOUGHTLENS_COLORS",
    "heatmap_colorscale",
    "node_color",
    "edge_color",
    "cosine_sim",
    "l2_normalise",
    "pca_2d",
    "softmax",
    "topk_indices",
    "topk_mask",
    "MASK_TOKEN",
    "mask_positions",
    "replace_token",
    "token_join",
    "whitespace_tokens",
]
