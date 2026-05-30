# ADR-0005: The proxy returns upstream responses verbatim

- Status: Accepted
- Date: 2026-05-29

## Context

To act as a "middle layer", ThoughtLens runs an OpenAI/Anthropic-compatible
proxy. Users sometimes expect it to transparently intercept any app — including
closed desktop apps. That is neither possible (TLS, hardcoded endpoints) nor
desirable (it would alter app behavior and violate ToS).

## Decision

The proxy forwards each request to the configured upstream and returns the
response **byte-for-byte**; it only *observes* (tees a copy to the dashboard).
Deep attribution (token masking) is opt-in, so the proxy never silently
multiplies the user's API spend. Only apps that are *configured* to use the
proxy base URL flow through it.

## Consequences

- The calling app's behavior and safety are never modified.
- Closed apps that hardcode their endpoint cannot be intercepted — documented as
  a deliberate boundary in the UI and docs.
- Passive observation uses real logprobs at zero extra cost; masking is explicit.
