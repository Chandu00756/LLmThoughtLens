"""Tests for the live white-box streams: ``server/whitebox_stream.py`` and ``server/xray.py``.

The heavy model load is replaced by an offline tiny GPT-2 (see ``_tiny_hf``)
by patching :meth:`HuggingFaceProvider._load`, and the global event bus is
swapped for a recorder, so every published number can be checked against an
independent forward pass of the same model.  The tiny model's norms are
perturbed so a double-applied final norm would change the numbers (a fresh
LayerNorm is idempotent and would hide it); the reference reads the true
residual stream with its own forward hooks on the blocks.  The HTTP routes are exercised
with FastAPI's TestClient against a minimal app containing only these routers
(no config files are touched).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from LLmThoughtLens.server import whitebox_stream, xray  # noqa: E402

from _tiny_hf import N_LAYER, WordTokenizer, install_tiny_loader, make_tiny_gpt2  # noqa: E402

PROMPT = "the capital of France is"


class _RecorderBus:
    """Stand-in for :class:`EventBus` that records ``(kind, payload)`` pairs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.got_event = threading.Event()

    def publish(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.events.append((kind, payload or {}))
        self.got_event.set()
        return {"kind": kind, "data": payload or {}}

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [d for k, d in self.events if k == kind]


@pytest.fixture
def bus(monkeypatch) -> _RecorderBus:
    rec = _RecorderBus()
    monkeypatch.setattr(whitebox_stream, "get_bus", lambda: rec)
    monkeypatch.setattr(xray, "get_bus", lambda: rec)
    return rec


def _greedy_reference(prompt: str, n_steps: int, seed: int = 0) -> list[dict[str, Any]]:
    """Independent greedy decode of the tiny model, recording per-step internals.

    The residual stream is every block's output (captured with plain forward
    hooks), i.e. *before* ``ln_f`` on the last layer too.
    """
    model = make_tiny_gpt2(seed, perturb_norms=True)
    tok = WordTokenizer()
    ids = torch.tensor([tok.encode_ids(prompt)])
    steps: list[dict[str, Any]] = []
    captured: list[Any] = []
    handles = [
        b.register_forward_hook(
            lambda _m, _a, o: captured.append(o[0] if isinstance(o, tuple) else o)
        )
        for b in model.transformer.h
    ]
    with torch.no_grad():
        for _ in range(n_steps):
            captured.clear()
            out = model(input_ids=ids, use_cache=False)
            hidden = list(captured)
            probs = torch.softmax(out.logits[0, -1], dim=-1)
            top_p, top_i = torch.topk(probs, k=5)
            final_norm = model.transformer.ln_f
            unembed = model.get_output_embeddings()
            lens = []
            for h in hidden:
                p_l = torch.softmax(unembed(final_norm(h[0, -1])), dim=-1)
                lens.append(torch.topk(p_l, k=3).indices.tolist())
            # The last layer's lens IS the model's prediction (final norm applied once).
            assert lens[-1][0] == int(top_i[0])
            steps.append(
                {
                    "next_id": int(top_i[0]),
                    "top_ids": top_i.tolist(),
                    "top_p": top_p.tolist(),
                    "layer_norms": [float(h[0, -1].norm()) for h in hidden],
                    "grid": [[float(h[0, t].norm()) for t in range(ids.shape[1])] for h in hidden],
                    "lens_ids": lens,
                    "seq": ids.shape[1],
                }
            )
            ids = torch.cat([ids, top_i[:1].view(1, 1)], dim=1)
    for h in handles:
        h.remove()
    return steps


# ===========================================================================
# White-box "thinking stream"
# ===========================================================================


class TestStreamWhitebox:
    def test_publishes_real_per_layer_norms_then_full_trace(self, monkeypatch, bus):
        loaded = install_tiny_loader(monkeypatch, perturb_norms=True)
        n = 3
        req = whitebox_stream.WhiteboxRequest(
            model_name="tiny-gpt2", prompt=PROMPT, max_new_tokens=n, device="cpu"
        )
        ret = whitebox_stream._stream_whitebox(req)

        assert bus.kinds() == (
            ["whitebox_started"] + ["whitebox_step"] * n + ["trace_complete", "whitebox_complete"]
        )
        started = bus.of("whitebox_started")[0]
        assert started == {"model": "tiny-gpt2", "device": "cpu", "prompt": PROMPT}

        ref = _greedy_reference(PROMPT, n)
        for i, (step, exp) in enumerate(zip(bus.of("whitebox_step"), ref, strict=True)):
            assert step["step"] == i
            assert step["n_layers"] == N_LAYER
            assert step["token"] == f"<{exp['next_id']}>"
            assert step["layer_norms"] == pytest.approx(exp["layer_norms"], rel=1e-4)
            assert [t for t, _ in step["top_tokens"]] == [f"<{j}>" for j in exp["top_ids"]]
            assert [p for _, p in step["top_tokens"]] == pytest.approx(exp["top_p"], rel=1e-4)

        expected_completion = "".join(f"<{s['next_id']}>" for s in ref)
        assert ret == {"completion": expected_completion, "n_steps": n}
        assert bus.of("whitebox_complete")[0] == ret

        trace = bus.of("trace_complete")[0]
        assert trace["completion"] == expected_completion
        assert trace["provider"] == "huggingface"
        assert trace["model"] == "hf/tiny-gpt2"
        assert trace["evidence_kind"] == "white_box"
        assert trace["prompt"] == PROMPT
        assert trace["features"] and trace["graph"]["nodes"]
        assert len(loaded) == 1  # the model is loaded exactly once

    def test_stops_at_eos(self, monkeypatch, bus):
        first_id = _greedy_reference(PROMPT, 1)[0]["next_id"]
        install_tiny_loader(monkeypatch, perturb_norms=True, eos_token_id=first_id)
        req = whitebox_stream.WhiteboxRequest(
            model_name="tiny-gpt2", prompt=PROMPT, max_new_tokens=10, device="cpu"
        )
        ret = whitebox_stream._stream_whitebox(req)
        assert ret == {"completion": f"<{first_id}>", "n_steps": 1}
        assert bus.kinds().count("whitebox_step") == 1

    def test_zero_tokens_gives_empty_completion(self, monkeypatch, bus):
        install_tiny_loader(monkeypatch)
        req = whitebox_stream.WhiteboxRequest(
            model_name="tiny-gpt2", prompt=PROMPT, max_new_tokens=0, device="cpu"
        )
        assert whitebox_stream._stream_whitebox(req) == {"completion": "", "n_steps": 0}
        assert "whitebox_step" not in bus.kinds()
        assert bus.of("trace_complete")[0]["completion"] == ""

    def test_request_defaults(self):
        req = whitebox_stream.WhiteboxRequest()
        assert req.prompt and req.model_name
        assert req.max_new_tokens > 0
        assert req.device == "auto"


# ===========================================================================
# Live LLM X-ray (logit lens)
# ===========================================================================


class TestStreamXray:
    def test_emits_logit_lens_grid_and_attention_from_real_forward(self, monkeypatch, bus):
        install_tiny_loader(monkeypatch, perturb_norms=True)
        n = 2
        req = xray.XrayRequest(
            model_name="tiny-gpt2", prompt=PROMPT, max_new_tokens=n, device="cpu"
        )
        ret = xray._stream_xray(req)

        assert bus.kinds() == ["xray_started"] + ["xray_step"] * n + ["xray_complete"]
        started = bus.of("xray_started")[0]
        assert started["model"] == "tiny-gpt2"
        assert started["prompt"] == PROMPT
        assert started["has_logit_lens"] is True
        assert started["has_final_norm"] is True

        ref = _greedy_reference(PROMPT, n)
        for step, exp in zip(bus.of("xray_step"), ref, strict=True):
            seq = exp["seq"]
            assert step["n_layers"] == N_LAYER
            assert step["token"] == f"<{exp['next_id']}>"
            assert len(step["tokens"]) == seq
            # Logit lens: per-layer top-3 through the real ln_f + lm_head.
            assert [ll["layer"] for ll in step["logit_lens"]] == list(range(N_LAYER))
            for ll, exp_ids in zip(step["logit_lens"], exp["lens_ids"], strict=True):
                assert [t for t, _ in ll["top"]] == [f"<{j}>" for j in exp_ids]
                probs = [p for _, p in ll["top"]]
                assert probs == sorted(probs, reverse=True)
            # Regression (double final norm): the last layer's top-1 is the token
            # the model actually emits, with the model's own probability.
            assert step["logit_lens"][-1]["top"][0][0] == step["token"]
            assert step["logit_lens"][-1]["top"][0][1] == pytest.approx(exp["top_p"][0], rel=1e-4)
            # Residual-stream magnitude grid (n_layers x seq).
            assert len(step["grid"]) == N_LAYER
            for row, exp_row in zip(step["grid"], exp["grid"], strict=True):
                assert row == pytest.approx(exp_row, rel=1e-4)
            # Head-averaged last-layer attention: causal, rows sum to 1.
            attn = step["attention"]
            assert len(attn) == seq and all(len(r) == seq for r in attn)
            for i, r in enumerate(attn):
                assert sum(r) == pytest.approx(1.0, abs=1e-4)
                assert all(v == pytest.approx(0.0, abs=1e-6) for v in r[i + 1 :])

        expected = "".join(f"<{s['next_id']}>" for s in ref)
        assert ret == {"completion": expected, "n_steps": n}
        assert bus.of("xray_complete")[0] == ret

    def test_xray_request_defaults(self):
        req = xray.XrayRequest()
        assert req.max_new_tokens > 0
        assert req.device == "auto"


# ===========================================================================
# HTTP routes (background task kick-off + error reporting)
# ===========================================================================


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(whitebox_stream.build_router())
    a.include_router(xray.build_router())
    return a


class TestRoutes:
    @pytest.mark.parametrize(
        ("module", "fn_name", "path"),
        [
            (whitebox_stream, "_stream_whitebox", "/api/whitebox/stream"),
            (xray, "_stream_xray", "/api/xray/stream"),
        ],
    )
    def test_post_starts_background_run(self, app, bus, monkeypatch, module, fn_name, path):
        seen: list[Any] = []
        done = threading.Event()

        def _fake(req: Any) -> dict[str, Any]:
            seen.append(req)
            done.set()
            return {"completion": "", "n_steps": 0}

        monkeypatch.setattr(module, fn_name, _fake)
        with TestClient(app) as client:
            r = client.post(path, json={"model_name": "tiny-gpt2", "prompt": "hi there"})
            assert r.status_code == 200
            assert r.json() == {"started": True, "model": "tiny-gpt2"}
            assert done.wait(5.0), "background worker never ran"
        assert seen[0].prompt == "hi there"
        assert seen[0].model_name == "tiny-gpt2"

    @pytest.mark.parametrize(
        ("module", "fn_name", "path", "event"),
        [
            (whitebox_stream, "_stream_whitebox", "/api/whitebox/stream", "whitebox_error"),
            (xray, "_stream_xray", "/api/xray/stream", "xray_error"),
        ],
    )
    def test_worker_failure_is_published_not_raised(
        self, app, bus, monkeypatch, module, fn_name, path, event
    ):
        def _boom(_req: Any) -> dict[str, Any]:
            raise OSError("weights not found")

        monkeypatch.setattr(module, fn_name, _boom)
        with TestClient(app) as client:
            r = client.post(path, json={})
            assert r.status_code == 200
            assert bus.got_event.wait(5.0), "error event never published"
        assert bus.of(event) == [{"error": "OSError: weights not found"}]

    def test_invalid_body_is_rejected(self, app):
        with TestClient(app) as client:
            r = client.post("/api/xray/stream", json={"max_new_tokens": "lots"})
        assert r.status_code == 422
