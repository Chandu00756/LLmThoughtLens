"""Quickstart example — demonstrates llmscope with the built-in MockProvider.

Run with:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from llmscope import Scope, Feature, FeatureSet, AttributionGraph
from llmscope.providers.registry import list_providers

# ------------------------------------------------------------------
# 1. Create a scope backed by the deterministic MockProvider
# ------------------------------------------------------------------
print("Available providers:", list_providers())

scope = Scope.from_mock(n_layers=4, n_heads=2, d_model=32, seed=42)
print("\nScope:", scope)

# ------------------------------------------------------------------
# 2. Trace a prompt
# ------------------------------------------------------------------
prompt = "The capital of France is Paris"
out = scope.trace(prompt)

print(f"\nPrompt  : {out.prompt!r}")
print(f"Tokens  : {out.tokens}")
print(f"Token IDs: {out.token_ids}")
print(f"Activations shape : {out.activations.shape}")  # type: ignore[union-attr]
print(f"Attentions shape  : {out.attentions.shape}")   # type: ignore[union-attr]
print(f"Logits shape      : {out.logits.shape}")       # type: ignore[union-attr]
print(f"Top 5 next tokens : {out.top_tokens}")

# ------------------------------------------------------------------
# 3. Work with Features and FeatureSets
# ------------------------------------------------------------------
fs = FeatureSet(name="capital-city-circuit")
for i, (tok, score) in enumerate(out.top_tokens[:5]):
    fs.add(Feature(id=i, label=f"feature:{tok}", layer=2, score=float(score)))

print(f"\nFeatureSet: {fs}")
print("Top 3 features:")
for f in fs.top(3):
    print(f"  {f}")

# ------------------------------------------------------------------
# 4. Build an AttributionGraph
# ------------------------------------------------------------------
graph = AttributionGraph(name="quickstart-graph")
for edge_src, edge_dst, weight in [(0, 1, 0.9), (1, 2, 0.7), (0, 2, 0.4)]:
    graph.add_edge(edge_src, edge_dst, weight=weight)

print(f"\n{graph}")
print(f"  Successors of 0 : {graph.successors(0)}")
print(f"  Predecessors of 2: {graph.predecessors(2)}")

# ------------------------------------------------------------------
# 5. Determinism check
# ------------------------------------------------------------------
scope2 = Scope.from_mock(n_layers=4, n_heads=2, d_model=32, seed=42)
out2 = scope2.trace(prompt)
assert np.array_equal(out.activations, out2.activations), "Activations should be reproducible"
print("\nDeterminism check passed — same seed, same activations.")
