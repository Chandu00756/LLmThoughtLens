"""Tests for the live LLM X-ray endpoint plumbing.

The heavy generation loop needs torch + a real model (covered by live runs),
but the final-norm resolver and the route wiring are pure and tested here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from LLmThoughtLens.server.app import create_app  # noqa: E402
from LLmThoughtLens.server.xray import _resolve_final_norm  # noqa: E402


def _obj(**attrs):
    o = type("Obj", (), {})()
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


class TestFinalNormResolver:
    def test_gpt2_path(self):
        model = _obj(transformer=_obj(ln_f="NORM"))
        assert _resolve_final_norm(model) == "NORM"

    def test_llama_path(self):
        model = _obj(model=_obj(norm="RMS"))
        assert _resolve_final_norm(model) == "RMS"

    def test_gptneox_path(self):
        model = _obj(gpt_neox=_obj(final_layer_norm="LN"))
        assert _resolve_final_norm(model) == "LN"

    def test_none_when_absent(self):
        assert _resolve_final_norm(_obj(unrelated=1)) is None


class TestRouteRegistered:
    def test_xray_route_present(self):
        app = create_app()
        paths = {r.path for r in app.routes}
        assert "/api/xray/stream" in paths


class TestRunXrayLoopGuardsEmptyAttentions:
    """Regression: a model whose attention backend returns an EMPTY tuple (e.g.
    SDPA) must not crash the X-ray loop — attention degrades to []."""

    def test_empty_attentions_tuple_does_not_crash(self):
        torch = pytest.importorskip("torch")
        from LLmThoughtLens.xray_core import run_xray_loop

        n_layers, d, vocab = 3, 4, 6

        class _Batch(dict):
            def to(self, _device):
                return self

        class _Tok:
            eos_token_id = None

            def __call__(self, _prompt, return_tensors=None):
                return _Batch(input_ids=torch.tensor([[1, 2, 3]]))

            def decode(self, ids):
                return "x"

        class _Out:
            def __init__(self, seq):
                self.hidden_states = tuple(torch.randn(1, seq, d) for _ in range(n_layers + 1))
                self.logits = torch.randn(1, seq, vocab)
                self.attentions = ()  # the empty-tuple case that used to crash

        class _Model:
            def get_output_embeddings(self):
                return torch.nn.Linear(d, vocab)

            def __call__(self, input_ids=None, **_kw):
                return _Out(input_ids.shape[1])

        events: list[tuple[str, dict]] = []
        run_xray_loop(
            _Model(),
            _Tok(),
            torch.device("cpu"),
            "hi there",
            2,
            emit=lambda k, data: events.append((k, data)),
            model_label="fake",
        )
        kinds = [k for k, _ in events]
        assert "xray_started" in kinds and "xray_complete" in kinds
        steps = [d for k, d in events if k == "xray_step"]
        assert steps, "no xray_step emitted"
        # Logit lens present + real shapes; attention safely empty.
        assert steps[0]["attention"] == []
        assert len(steps[0]["logit_lens"]) == n_layers
        assert len(steps[0]["grid"]) == n_layers
