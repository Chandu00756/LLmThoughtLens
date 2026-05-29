"""MockProvider — deterministic, dependency-free provider for tests + examples.

Returns real-shaped arrays (activations, attentions, logits, top-token list)
seeded from ``(seed, prompt)`` so that every call with the same inputs gives
byte-identical outputs.  Used to satisfy Phase 1's exit criterion: a
mock-based smoke test that runs end-to-end without any API key.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from thoughtlens.providers.base import BaseProvider, ProviderOutput
from thoughtlens.utils.math_utils import softmax
from thoughtlens.utils.tokenizer_utils import whitespace_tokens

# Small, fixed vocabulary used both for token_ids and for surface forms of
# top-k next-token predictions.  Real backends supply their own ids/strings.
_FAKE_VOCAB: tuple[str, ...] = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "the",
    "a",
    "is",
    "of",
    "to",
    "and",
    "in",
    "that",
    "it",
    "was",
    "for",
    "on",
    "are",
    "as",
    "with",
    "by",
    "from",
    "this",
    "his",
    "her",
    "they",
    "we",
    "I",
    "be",
    "at",
    "one",
    "have",
    "Hello",
    "world",
    "Austin",
    "Texas",
    "Paris",
    "France",
    "capital",
    "city",
    "state",
    "Dallas",
    "containing",
    "river",
    "ocean",
    "name",
    "country",
    "mountain",
    "language",
    "is.",
    ".",
    ",",
)


class MockProvider(BaseProvider):
    """Deterministic mock backend — no network, no GPU, no API key required.

    Parameters
    ----------
    n_layers, n_heads, d_model:
        Shape of the simulated transformer.  Sensible defaults keep memory
        usage tiny so the mock is usable in CI.
    vocab_size:
        Defaults to the size of the built-in fake vocabulary.
    seed:
        Integer seed; the actual RNG mixes this with ``hash(prompt)`` so
        different prompts produce different outputs while the same prompt
        is reproducible.
    """

    evidence_kind = "white_box"

    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 4,
        d_model: int = 64,
        vocab_size: int = len(_FAKE_VOCAB),
        seed: int = 42,
    ) -> None:
        if n_layers <= 0 or n_heads <= 0 or d_model <= 0 or vocab_size <= 0:
            raise ValueError("n_layers, n_heads, d_model, vocab_size must all be > 0")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # BaseProvider API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return f"mock-L{self.n_layers}-H{self.n_heads}-D{self.d_model}"

    def run(self, prompt: str, **_: Any) -> ProviderOutput:
        rng = self._rng_for(prompt)
        tokens = whitespace_tokens(prompt)
        n_tokens = len(tokens)

        token_ids = [abs(hash(t)) % self.vocab_size for t in tokens]

        activations = self._activations(rng, n_tokens)
        attentions = self._attentions(rng, n_tokens)
        logits = self._logits(rng, n_tokens)

        last_logits = logits[-1]
        probs = softmax(last_logits)
        top_k = min(5, self.vocab_size)
        top_idx = np.argpartition(-probs, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-probs[top_idx])]
        top_tokens: list[tuple[str, float]] = [
            (
                _FAKE_VOCAB[i] if i < len(_FAKE_VOCAB) else f"tok_{i}",
                float(probs[i]),
            )
            for i in top_idx.tolist()
        ]

        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=token_ids,
            activations=activations,
            attentions=attentions,
            logits=last_logits.astype(np.float32),
            top_tokens=top_tokens,
            evidence_kind="white_box",
            meta={
                "provider": "MockProvider",
                "model": self.model_id,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_model": self.d_model,
                "vocab_size": self.vocab_size,
                "seed": self.seed,
                "all_logits": logits.astype(np.float32),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rng_for(self, prompt: str) -> np.random.Generator:
        mix = (self.seed * 0x9E3779B1) ^ (abs(hash(prompt)) & 0x7FFFFFFF)
        return np.random.default_rng(mix & 0xFFFFFFFF)

    def _activations(self, rng: np.random.Generator, n_tokens: int) -> np.ndarray:
        # Make activations grow slightly with depth so layer-PCA shows a real
        # trajectory instead of an isotropic Gaussian blob.
        base = rng.standard_normal((self.n_layers, n_tokens, self.d_model)).astype(np.float32)
        layer_gain = np.linspace(1.0, 1.5, self.n_layers, dtype=np.float32)
        base *= layer_gain[:, None, None]
        return base

    def _attentions(self, rng: np.random.Generator, n_tokens: int) -> np.ndarray:
        raw = rng.standard_normal((self.n_layers, self.n_heads, n_tokens, n_tokens)).astype(np.float32)
        # Causal mask: upper triangle (future positions) set to -inf before softmax.
        causal = np.full((n_tokens, n_tokens), -np.inf, dtype=np.float32)
        causal = np.triu(causal, k=1)
        raw = raw + causal[None, None, :, :]
        return softmax(raw, axis=-1)

    def _logits(self, rng: np.random.Generator, n_tokens: int) -> np.ndarray:
        return rng.standard_normal((n_tokens, self.vocab_size)).astype(np.float32)
