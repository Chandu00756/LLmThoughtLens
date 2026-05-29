"""Tests for FeatureExtractor — real masking + SAE feature paths."""

from __future__ import annotations

import numpy as np
from LLmThoughtLens.features.extractor import FeatureExtractor
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider
from LLmThoughtLens.utils.tokenizer_utils import MASK_TOKEN


class TestWhiteBox:
    def test_returns_white_box_features(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=1)
        out = mp.run("hello world test")
        feats = FeatureExtractor(top_k=8).extract(out, provider=mp)
        assert len(feats) == 8
        assert all(f.evidence_kind == "white_box" for f in feats)
        # Sorted by score descending
        scores = [f.score for f in feats]
        assert scores == sorted(scores, reverse=True)

    def test_score_is_l2_norm(self):
        mp = MockProvider(n_layers=2, n_heads=2, d_model=8, seed=5)
        out = mp.run("a b")
        feats = FeatureExtractor(top_k=4).extract(out, provider=mp)
        for f in feats:
            expected = float(np.linalg.norm(out.activations[f.layer, f.token_idx]))
            assert abs(f.score - expected) < 1e-5


class _BlackBoxWrapper(BaseProvider):
    """Strips activations so the extractor takes the black-box path.

    Returns deterministic top-token probabilities so masking has a real effect.
    """

    evidence_kind = "black_box"

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "bb_test"

    def run(self, prompt: str, **_) -> ProviderOutput:
        self.calls.append(prompt)
        tokens = prompt.split() or ["<empty>"]
        # Prob of "X" is high when "Y" is unmasked in the prompt, else low.
        prob_x = 0.9 if "Y" in tokens else 0.1
        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=[],
            top_tokens=[("X", prob_x), ("Z", 1.0 - prob_x)],
            evidence_kind="black_box",
        )


class TestBlackBox:
    def test_masking_produces_real_importance_score(self):
        bb = _BlackBoxWrapper()
        ex = FeatureExtractor(top_k=5, blackbox_budget=4)
        out = bb.run("A Y B C")
        feats = ex.extract(out, provider=bb)
        # The 'Y' token should have the highest causal importance score.
        y_feat = next(f for f in feats if f.label.endswith("Y"))
        other = [f for f in feats if not f.label.endswith("Y")]
        for f in other:
            assert y_feat.score > f.score, f"Y importance ({y_feat.score}) not > other ({f.score})"
        assert all(f.evidence_kind == "black_box" for f in feats)
        assert all(f.meta.get("method") == "token_masking" for f in feats)

    def test_mask_tokens_were_used(self):
        bb = _BlackBoxWrapper()
        ex = FeatureExtractor(top_k=5, blackbox_budget=3)
        out = bb.run("A B C")
        ex.extract(out, provider=bb)
        # At least one provider call masked a token.
        assert any(MASK_TOKEN in call for call in bb.calls)

    def test_pairwise_interactions(self):
        bb = _BlackBoxWrapper()
        ex = FeatureExtractor(top_k=5, blackbox_budget=3)
        pairs = ex.compute_pairwise_interactions(bb, "A Y B", budget=3)
        # n=3 → 3 unordered pairs.
        assert len(pairs) == 3
        for (i, j), score in pairs.items():
            assert i < j
            assert isinstance(score, float)
