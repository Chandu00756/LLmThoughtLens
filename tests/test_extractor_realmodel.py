"""Real-model checks for white-box feature extraction (GPT-2 family, offline only).

GPT-2 puts a massive activation ("attention sink") on the first position:
its residual norm is 10-50x every other token at nearly every layer.  These
tests prove the default extractor ranks token-specific signal instead of
that sink.  They never download anything: weights are loaded with
``local_files_only=True`` and the whole module is skipped when they are not
already in the local HuggingFace cache.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from LLmThoughtLens.circuits.tracer import CircuitTracer
from LLmThoughtLens.features.extractor import FeatureExtractor
from LLmThoughtLens.providers.base import ProviderOutput

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_PROMPT = "The capital of the state containing Dallas is"
_CONTENT_WORDS = {" Dallas", " capital", " state"}


def _load_local(model_name: str) -> tuple[Any, Any]:
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:  # OSError when not cached; anything else = unusable here
        pytest.skip(f"{model_name} weights not available offline: {exc}")
    model.eval()
    return tok, model


def _white_box_output(tok: Any, model: Any, prompt: str) -> ProviderOutput:
    """Mirror HuggingFaceProvider's packing: hidden_states[1:] -> (L, T, D)."""
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = torch.stack(out.hidden_states[1:], dim=0).squeeze(1).to(torch.float32).numpy()
    ids = enc["input_ids"][0].tolist()
    probs = torch.softmax(out.logits[0, -1], dim=-1)
    top_id = int(torch.argmax(probs))
    return ProviderOutput(
        prompt=prompt,
        tokens=[tok.decode([i]) for i in ids],
        token_ids=ids,
        activations=hs,
        top_tokens=[(tok.decode([top_id]), float(probs[top_id]))],
        evidence_kind="white_box",
    )


@pytest.fixture(scope="module", params=["gpt2", "distilgpt2"])
def real_output(request: pytest.FixtureRequest) -> ProviderOutput:
    tok, model = _load_local(request.param)
    return _white_box_output(tok, model, _PROMPT)


def test_first_position_is_detected_as_massive_activation(real_output: ProviderOutput) -> None:
    ex = FeatureExtractor(top_k=10)
    ex.extract(real_output)
    assert 0 in ex.last_outlier_positions
    assert 0 in ex.last_excluded_positions
    assert ex.last_outlier_stats[0]["max_ratio"] > 6.0
    # Only the sink — content tokens are never mistaken for massive activations.
    assert ex.last_excluded_positions == [0]


def test_legacy_l2_scoring_is_dominated_by_the_sink(real_output: ProviderOutput) -> None:
    feats = FeatureExtractor(top_k=4, scoring="l2").extract(real_output)
    assert all(f.token_idx == 0 for f in feats)


def test_default_top_features_point_at_content_tokens(real_output: ProviderOutput) -> None:
    feats = FeatureExtractor(top_k=10).extract(real_output)
    assert len(feats) == 10
    assert all(f.token_idx != 0 for f in feats)
    positions = {f.token_idx for f in feats}
    assert len(positions) >= 2, f"top-10 collapsed onto one position: {positions}"
    top_tokens = {real_output.tokens[f.token_idx] for f in feats}
    assert top_tokens & _CONTENT_WORDS, f"top features miss content words: {top_tokens}"
    for f in feats:
        expected = float(np.linalg.norm(real_output.activations[f.layer, f.token_idx]))
        assert f.meta["raw_norm"] == pytest.approx(expected, rel=1e-4)


def test_trace_graph_records_excluded_sink(real_output: ProviderOutput) -> None:
    feats = FeatureExtractor(top_k=20).extract(real_output)
    graph = CircuitTracer(min_weight=0.05).trace(real_output, feats)
    assert graph.meta["excluded_positions"] == [0]
    err = graph.node(CircuitTracer.error_node_id())
    assert err is not None
    assert 0.0 <= err.meta["unexplained_fraction"] <= 1.0
    # The sink carries the overwhelming majority of residual-stream energy.
    assert err.meta["excluded_energy_fraction"] > 0.5
