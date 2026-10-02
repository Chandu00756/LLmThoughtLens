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


class TestRunXrayLoopFallbackHasNoDoubleNorm:
    """Models whose blocks cannot be located use ``output_hidden_states``; the
    last entry is already final-normed (HF convention), so the lens must not
    normalise it again."""

    def test_final_norm_skipped_on_last_hidden_state(self):
        torch = pytest.importorskip("torch")
        from LLmThoughtLens.xray_core import run_xray_loop

        n_layers, d, vocab = 3, 4, 6
        norm_calls: list[int] = []

        def final_norm(h):
            norm_calls.append(1)
            return h * 100.0  # dramatic, so a double application is visible

        class _Batch(dict):
            def to(self, _device):
                return self

        class _Tok:
            eos_token_id = None

            def __call__(self, _prompt, return_tensors=None):
                return _Batch(input_ids=torch.tensor([[1, 2]]))

            def decode(self, ids):
                return f"<{ids[0]}>" if len(ids) == 1 else "x"

        unembed = torch.nn.Linear(d, vocab, bias=False)
        torch.nn.init.eye_(unembed.weight[:d])
        hidden = [torch.randn(1, 8, d) for _ in range(n_layers + 1)]

        class _Out:
            def __init__(self, seq):
                self.hidden_states = tuple(h[:, :seq] for h in hidden)
                self.logits = unembed(hidden[-1][:, :seq])
                self.attentions = None

        class _Model:
            # ``transformer.ln_f`` resolves as the final norm, but there are no blocks.
            transformer = type("T", (), {"ln_f": staticmethod(final_norm)})()

            def get_output_embeddings(self):
                return unembed

            def __call__(self, input_ids=None, **_kw):
                return _Out(input_ids.shape[1])

        events: list[tuple[str, dict]] = []
        run_xray_loop(_Model(), _Tok(), "cpu", "p", 1, emit=lambda k, v: events.append((k, v)))
        started = next(v for k, v in events if k == "xray_started")
        assert started["has_final_norm"] is True
        step = next(v for k, v in events if k == "xray_step")
        assert len(norm_calls) == n_layers - 1  # every layer but the last
        with torch.no_grad():
            expected = torch.softmax(unembed(hidden[-1][0, 1]), -1).max()
        assert step["logit_lens"][-1]["top"][0][1] == pytest.approx(float(expected), rel=1e-5)
        assert step["logit_lens"][-1]["top"][0][0] == step["token"]


class TestRunXrayLoopHooked:
    def test_hooked_argument_and_auto_wrap_agree(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from LLmThoughtLens.models import HookedModel
        from LLmThoughtLens.xray_core import run_xray_loop

        from _tiny_hf import WordTokenizer, make_tiny_gpt2

        model, tok = make_tiny_gpt2(perturb_norms=True), WordTokenizer()
        runs = []
        for hooked in (None, HookedModel(model, tok, device="cpu")):
            events: list[tuple[str, dict]] = []
            ret = run_xray_loop(
                model,
                tok,
                "cpu",
                "a b c",
                2,
                emit=lambda k, v, e=events: e.append((k, v)),
                hooked=hooked,
            )
            runs.append((ret, events))
        assert runs[0] == runs[1]
        steps = [v for k, v in runs[0][1] if k == "xray_step"]
        assert [s["logit_lens"][-1]["top"][0][0] for s in steps] == [s["token"] for s in steps]
        assert set(steps[0]) == {
            "step",
            "token",
            "tokens",
            "n_layers",
            "logit_lens",
            "grid",
            "attention",
        }

    def test_resolve_final_norm_is_the_shared_resolver(self):
        from LLmThoughtLens.models.families import resolve_final_norm
        from LLmThoughtLens.xray_core import resolve_final_norm as xray_resolver

        assert xray_resolver is resolve_final_norm is _resolve_final_norm

    def test_opt_path_and_family_first_resolution(self):
        assert _resolve_final_norm(_obj(model=_obj(decoder=_obj(final_layer_norm="OPT")))) == "OPT"
        cfg = _obj(model_type="gpt_neox")
        # Family-first: a GPT-NeoX config picks gpt_neox.final_layer_norm even if
        # a Llama-style ``model.norm`` attribute also exists.
        model = _obj(config=cfg, model=_obj(norm="WRONG"), gpt_neox=_obj(final_layer_norm="NEOX"))
        assert _resolve_final_norm(model) == "NEOX"
