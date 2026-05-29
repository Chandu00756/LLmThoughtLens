"""
MockProvider — deterministic, zero-dependency provider for tests and examples.

All numerical arrays are generated from a seeded RNG so that unit tests are
fully reproducible without any API keys or GPU access.
"""

from __future__ import annotations

import numpy as np

from llmscope.providers.base import BaseProvider, ProviderOutput

_FAKE_VOCAB: list[str] = [
    "<pad>", "<s>", "</s>", "the", "a", "is", "of", "to", "and", "in",
    "that", "it", "was", "he", "she", "we", "for", "on", "are", "as",
    "with", "his", "they", "I", "be", "at", "one", "have", "by", "from",
    "Hello", "world", "foo", "bar", "baz",
]


class MockProvider(BaseProvider):
    """Deterministic mock provider — no API keys, no network, no GPU.

    Generates synthetic tokens, activations, attentions, and logits from a
    seeded NumPy RNG so that every call with the same ``(prompt, seed)`` pair
    returns identical arrays.

    Parameters
    ----------
    n_layers:
        Number of simulated transformer layers.
    n_heads:
        Number of simulated attention heads per layer.
    d_model:
        Hidden dimension of the simulated model.
    vocab_size:
        Vocabulary size for logit generation.
    seed:
        Integer seed for the RNG — controls reproducibility.
    """

    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 4,
        d_model: int = 64,
        vocab_size: int = len(_FAKE_VOCAB),
        seed: int = 42,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seed = seed

    def run(self, prompt: str, **kwargs) -> ProviderOutput:
        """Run mock inference and return a deterministic :class:`ProviderOutput`.

        The token sequence is produced by splitting on whitespace; every call
        with the same prompt returns the same arrays.

        Parameters
        ----------
        prompt:
            The input text.
        **kwargs:
            Ignored (present for API compatibility).
        """
        rng = np.random.default_rng(self.seed + hash(prompt) % (2**31))

        tokens: list[str] = prompt.split() or ["<s>"]
        n_tokens = len(tokens)
        token_ids: list[int] = [
            abs(hash(tok)) % self.vocab_size for tok in tokens
        ]

        activations = rng.standard_normal(
            (self.n_layers, n_tokens, self.d_model)
        ).astype(np.float32)

        # Attention: softmax-normalised over key dimension
        raw_attn = rng.standard_normal(
            (self.n_layers, self.n_heads, n_tokens, n_tokens)
        ).astype(np.float32)
        attn_max = raw_attn.max(axis=-1, keepdims=True)
        exp_attn = np.exp(raw_attn - attn_max)
        attentions = exp_attn / exp_attn.sum(axis=-1, keepdims=True)

        # Logits for the last token position
        logits = rng.standard_normal((n_tokens, self.vocab_size)).astype(np.float32)

        # Top-5 next-token predictions (softmax over last-token logits)
        last_logits = logits[-1]
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()
        top_k = min(5, self.vocab_size)
        top_ids = np.argsort(probs)[-top_k:][::-1].tolist()
        top_tokens: list[tuple[str, float]] = [
            (_FAKE_VOCAB[i] if i < len(_FAKE_VOCAB) else f"tok_{i}", float(probs[i]))
            for i in top_ids
        ]

        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=token_ids,
            activations=activations,
            attentions=attentions,
            logits=logits,
            top_tokens=top_tokens,
            meta={
                "provider": "MockProvider",
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_model": self.d_model,
                "seed": self.seed,
            },
        )

    @property
    def name(self) -> str:
        return "mock"
