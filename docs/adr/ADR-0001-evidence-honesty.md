# ADR-0001: Evidence honesty — never synthesize internals

- Status: Accepted
- Date: 2026-05-29

## Context

ThoughtLens visualizes "what a model is doing." A tool like this is only useful
if every number it shows is real. The temptation is to fill gaps (e.g. when an
API does not expose activations) with plausible-looking synthetic data.

## Decision

Every `ProviderOutput` is tagged with an `evidence_kind`:

- `white_box` — activations/attentions were measured directly from the model.
- `black_box` — only API-level signals (logprobs, sampled tokens) are available.

`activations`, `attentions`, and `logits` are `None` whenever the backend cannot
legitimately produce them. They are **never** synthesized. The UI surfaces the
evidence kind on every trace and maps it to an observed / inferred / approximated
taxonomy so a reader never over-trusts a value.

## Consequences

- Black-box providers (OpenAI, Anthropic, Ollama) drive the attribution graph
  from real logprob deltas / token-masking perturbations, not fabricated
  activations.
- The mock provider is explicitly labelled synthetic and is for tests/demos only.
