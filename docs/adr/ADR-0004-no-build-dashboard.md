# ADR-0004: No-build SPA + FastAPI for the live dashboard

- Status: Accepted
- Date: 2026-05-29

## Context

The live dashboard must ship on PyPI and "just work" after `pip install`. A
React/Vite frontend would require a Node toolchain and a committed build bundle,
adding friction and supply-chain surface.

## Decision

The dashboard is a single-page app written in vanilla HTML/CSS/JS (Plotly via
CDN) served directly from the wheel by a FastAPI app. Live updates flow over a
WebSocket fed by an in-process `EventBus`. The `server` extra pulls only
`fastapi`, `uvicorn`, `websockets`, `sse-starlette`.

## Consequences

- Zero Node/build step; the static assets are packaged in the wheel.
- The same FastAPI app hosts the REST API, WebSocket, and the provider proxy.
- Styling is hand-written CSS with real transitions — no component framework.
