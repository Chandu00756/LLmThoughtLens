"""Offline tiny-HF fixtures shared by the white-box provider / stream / X-ray tests.

Nothing here downloads weights: tiny models are built from transformers
configs with a fixed torch seed, and :class:`WordTokenizer` is a deterministic
word-level tokenizer that implements exactly the surface the providers call
(``__call__(..., return_tensors="pt")`` -> batch with ``.to()``, ``decode``,
``eos_token_id`` / ``pad_token_id``, optional ``chat_template``).

* :func:`make_tiny_gpt2` / :func:`make_provider` / :func:`install_tiny_loader`
  — the 2-layer GPT-2 used across the suite.
* :func:`make_tiny_family_model` — a 2-layer random model for every family
  :class:`~LLmThoughtLens.models.HookedModel` supports (see
  :data:`TINY_FAMILIES`).

``perturb_norms=True`` randomises every norm's affine weights.  Freshly
initialised LayerNorms are the identity affine map, which makes
``norm(norm(x)) == norm(x)`` and hides double-normalisation bugs; perturbed
norms expose them.

Callers must ``pytest.importorskip("torch")`` / ``("transformers")`` before
importing this module.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

N_LAYER = 2
N_HEAD = 2
N_EMBD = 32
VOCAB = 64


class _Batch(dict):
    """dict that mimics ``transformers.BatchEncoding.to``."""

    def to(self, device: Any) -> _Batch:
        return _Batch({k: v.to(device) for k, v in self.items()})


class WordTokenizer:
    """Deterministic whitespace tokenizer over a ``VOCAB``-sized id space.

    Ids 0 and 1 are reserved for EOS / BOS; every word maps to a stable id in
    ``[2, VOCAB)`` and decodes to ``"<id>"``.  With ``chat_template`` set,
    :meth:`apply_chat_template` renders ``"<user> {content} <assistant>"``
    per message (the template string itself is only a presence flag).
    """

    def __init__(
        self,
        eos_token_id: int | None = 0,
        pad_token_id: int | None = 0,
        chat_template: str | None = None,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token = "<eos>"
        self.pad_token: str | None = None
        self.chat_template = chat_template
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def word_id(word: str) -> int:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()
        return 2 + int.from_bytes(digest, "big") % (VOCAB - 2)

    def encode_ids(self, prompt: str) -> list[int]:
        return [self.word_id(w) for w in prompt.split()] or [1]

    def __call__(self, prompt: str, return_tensors: str | None = None, **kwargs: Any) -> _Batch:
        self.calls.append({"prompt": prompt, **kwargs})
        ids = torch.tensor([self.encode_ids(prompt)], dtype=torch.long)
        return _Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids: Any) -> str:
        return "".join(f"<{int(i)}>" for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_kw: Any,
    ) -> str:
        assert not tokenize, "WordTokenizer only renders templates as text"
        text = " ".join(f"<{m['role']}> {m['content']}" for m in messages)
        return text + (" <assistant>" if add_generation_prompt else "")


def perturb_norm_weights(model: Any, seed: int = 0, scale: float = 0.5) -> Any:
    """Give every norm module non-trivial affine weights (in place); returns *model*."""
    gen = torch.Generator().manual_seed(seed + 1000)
    with torch.no_grad():
        for mod in model.modules():
            if "norm" not in type(mod).__name__.lower():
                continue
            for p in mod.parameters(recurse=False):
                p.add_(torch.randn(p.shape, generator=gen) * scale)
    return model


def make_tiny_gpt2(seed: int = 0, perturb_norms: bool = False) -> Any:
    """Return a randomly initialised 2-layer GPT-2 (eager attention, eval mode)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        vocab_size=VOCAB,
        n_positions=64,
        bos_token_id=1,
        eos_token_id=0,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(cfg).eval()
    return perturb_norm_weights(model, seed) if perturb_norms else model


def make_provider(
    capture_internals: bool = True, seed: int = 0, perturb_norms: bool = False, **kwargs: Any
) -> Any:
    """A HuggingFaceProvider with the tiny model injected so ``_load`` is a no-op."""
    from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

    provider = HuggingFaceProvider(
        model_name="tiny-gpt2", device="cpu", capture_internals=capture_internals
    )
    provider._model = make_tiny_gpt2(seed, perturb_norms=perturb_norms)
    provider._tokenizer = WordTokenizer(**kwargs)
    provider._device = "cpu"
    return provider


def install_tiny_loader(
    monkeypatch: Any, seed: int = 0, perturb_norms: bool = False, **tok_kwargs: Any
) -> list[Any]:
    """Patch ``HuggingFaceProvider._load`` to inject the tiny model offline.

    Returns a list that collects every provider instance that was loaded, so
    tests can inspect the exact model / tokenizer the code under test used.
    """
    from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

    loaded: list[Any] = []

    def _fake_load(self: Any) -> None:
        if self._model is not None:
            return
        self._model = make_tiny_gpt2(seed, perturb_norms=perturb_norms)
        self._tokenizer = WordTokenizer(**tok_kwargs)
        self._device = "cpu"
        loaded.append(self)

    monkeypatch.setattr(HuggingFaceProvider, "_load", _fake_load)
    return loaded


# ---------------------------------------------------------------------------
# Every supported family, tiny and random
# ---------------------------------------------------------------------------

_DECODER_COMMON: dict[str, Any] = {
    "num_hidden_layers": N_LAYER,
    "num_attention_heads": N_HEAD,
    "hidden_size": N_EMBD,
    "intermediate_size": 64,
    "vocab_size": VOCAB,
    "max_position_embeddings": 64,
    "bos_token_id": 1,
    "eos_token_id": 0,
    "pad_token_id": 0,
}

#: family name -> (transformers config class name, config kwargs)
TINY_FAMILIES: dict[str, tuple[str, dict[str, Any]]] = {
    "gpt2": (
        "GPT2Config",
        {
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
            "vocab_size": VOCAB,
            "n_positions": 64,
            "bos_token_id": 1,
            "eos_token_id": 0,
        },
    ),
    "gpt_neox": ("GPTNeoXConfig", dict(_DECODER_COMMON)),
    "llama": ("LlamaConfig", {**_DECODER_COMMON, "num_key_value_heads": N_HEAD}),
    "mistral": ("MistralConfig", {**_DECODER_COMMON, "num_key_value_heads": 1}),
    "qwen2": ("Qwen2Config", {**_DECODER_COMMON, "num_key_value_heads": 1}),
    "qwen3": ("Qwen3Config", {**_DECODER_COMMON, "num_key_value_heads": 1, "head_dim": 16}),
    "gemma": ("GemmaConfig", {**_DECODER_COMMON, "num_key_value_heads": 1, "head_dim": 16}),
    "gemma2": (
        "Gemma2Config",
        {
            **_DECODER_COMMON,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "final_logit_softcapping": 3.0,  # small cap so soft-capping really bites
        },
    ),
    "gemma3": (
        "Gemma3TextConfig",
        {**_DECODER_COMMON, "num_key_value_heads": 1, "head_dim": 16},
    ),
    "phi3": ("Phi3Config", {**_DECODER_COMMON, "num_key_value_heads": N_HEAD}),
    "opt": (
        "OPTConfig",
        {
            "num_hidden_layers": N_LAYER,
            "num_attention_heads": N_HEAD,
            "hidden_size": N_EMBD,
            "ffn_dim": 64,
            "vocab_size": VOCAB,
            "max_position_embeddings": 64,
            "word_embed_proj_dim": 16,  # != hidden_size -> exercises project_in/out
            "bos_token_id": 1,
            "eos_token_id": 0,
            "pad_token_id": 0,
        },
    ),
}


def make_tiny_family_model(family: str, seed: int = 0, perturb: bool = True) -> Any:
    """A random 2-layer model of *family* with eager attention, in eval mode.

    Returns ``None`` when the installed transformers lacks the family's
    config class.  ``perturb=True`` adds noise to every weight (norms
    included) so norms, soft-capping and projections are non-trivial.
    """
    import transformers
    from transformers import AutoModelForCausalLM

    cls_name, kwargs = TINY_FAMILIES[family]
    cfg_cls = getattr(transformers, cls_name, None)
    if cfg_cls is None:
        return None
    torch.manual_seed(seed)
    cfg = cfg_cls(**kwargs)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()
    if perturb:
        gen = torch.Generator().manual_seed(seed + 7)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=gen) * 0.2)
    return model
