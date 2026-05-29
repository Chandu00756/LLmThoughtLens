"""HuggingFace provider — runs transformer models locally via 🤗 Transformers."""

from __future__ import annotations

from llmscope.providers.base import BaseProvider, ProviderOutput


class HuggingFaceProvider(BaseProvider):
    """Provider that loads any 🤗 causal-LM model locally.

    Requires the ``huggingface`` extra:
    ``pip install 'llmscope[huggingface]'``.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"gpt2"`` or
        ``"meta-llama/Llama-3.2-1B"``.
    device:
        Target device string: ``"cpu"``, ``"cuda"``, ``"mps"``, …
        Defaults to ``"cpu"``.
    return_activations:
        When ``True``, hidden states and attentions are captured and packed
        into the :class:`~llmscope.providers.base.ProviderOutput`.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        return_activations: bool = True,
    ) -> None:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "HuggingFace provider requires the 'huggingface' extra. "
                "Install with: pip install 'llmscope[huggingface]'"
            ) from exc
        self.model_name = model_name
        self.device = device
        self.return_activations = return_activations
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer on first call."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_hidden_states=self.return_activations,
            output_attentions=self.return_activations,
        ).to(self.device)
        self._model.eval()

    def run(self, prompt: str, **kwargs) -> ProviderOutput:
        """Run inference and return a :class:`ProviderOutput`."""
        import torch
        import numpy as np

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        token_ids: list[int] = inputs["input_ids"][0].tolist()
        tokens: list[str] = [
            self._tokenizer.decode([tid]) for tid in token_ids
        ]

        with torch.no_grad():
            outputs = self._model(**inputs, **kwargs)

        activations = None
        attentions = None
        if self.return_activations and outputs.hidden_states:
            # Stack: (n_layers+1, n_tokens, d_model) → (n_layers, …)
            hs = torch.stack(outputs.hidden_states[1:], dim=0)
            activations = hs.squeeze(1).cpu().numpy().astype(np.float32)

        if self.return_activations and outputs.attentions:
            attn = torch.stack(outputs.attentions, dim=0)
            attentions = attn.squeeze(0).cpu().numpy().astype(np.float32)

        logits = outputs.logits.cpu().numpy().astype(np.float32)
        last_logits = logits[0, -1]
        probs = _softmax(last_logits)
        top_k = min(5, len(probs))
        top_ids = probs.argsort()[-top_k:][::-1].tolist()
        top_tokens: list[tuple[str, float]] = [
            (self._tokenizer.decode([i]), float(probs[i])) for i in top_ids
        ]

        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=token_ids,
            activations=activations,
            attentions=attentions,
            logits=logits[0],
            top_tokens=top_tokens,
            meta={"provider": "HuggingFaceProvider", "model": self.model_name},
        )

    @property
    def name(self) -> str:
        return f"hf/{self.model_name}"


def _softmax(x: "np.ndarray") -> "np.ndarray":
    import numpy as np

    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
