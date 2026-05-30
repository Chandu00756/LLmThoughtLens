# ADR-0002: A single provider envelope (ProviderOutput)

- Status: Accepted
- Date: 2026-05-29

## Context

ThoughtLens must work across closed APIs (OpenAI, Anthropic), local servers
(Ollama), and local weights (HuggingFace). Downstream code (feature extraction,
the circuit tracer, the report, the dashboard) should not branch on backend
identity.

## Decision

All backends implement `BaseProvider.run(prompt) -> ProviderOutput`. Every
downstream module reads only the uniform `ProviderOutput` fields
(`tokens`, `activations`, `attentions`, `logits`, `top_tokens`, `evidence_kind`,
`meta`). New providers are added behind a lazy registry so optional extras never
break `import LLmThoughtLens`.

## Consequences

- Adding a provider is a single new module + one registry loader.
- The same trace pipeline, report, and dashboard work for every backend.
- The proxy and SDK reuse the same envelope, so live traffic and offline traces
  render identically.
