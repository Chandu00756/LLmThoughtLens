"""llmscope CLI — command-line interface.

Entry point: ``llmscope`` (configured in pyproject.toml).

Commands
--------
trace   Run a prompt through a provider and print token / meta output.
probe   [stub] Run registered probes against cached activations.
version Show the llmscope version.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_trace(args: argparse.Namespace) -> None:
    """Handle the ``llmscope trace`` command."""
    provider_name: str = args.provider
    prompt: str = args.prompt

    if provider_name == "mock":
        from llmscope.providers.mock_provider import MockProvider

        provider = MockProvider(seed=args.seed)
    elif provider_name == "openai":
        from llmscope.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model=args.model)
    elif provider_name == "anthropic":
        from llmscope.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(model=args.model)
    elif provider_name == "ollama":
        from llmscope.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(model=args.model)
    else:
        print(f"Unknown provider: {provider_name}", file=sys.stderr)
        sys.exit(1)

    out = provider.run(prompt)

    if args.json:
        payload = {
            "prompt": out.prompt,
            "tokens": out.tokens,
            "token_ids": out.token_ids,
            "top_tokens": out.top_tokens,
            "meta": out.meta,
            "has_activations": out.activations is not None,
            "has_attentions": out.attentions is not None,
            "has_logits": out.logits is not None,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Provider : {out.meta.get('provider', provider_name)}")
        print(f"Tokens   : {out.tokens}")
        print(f"Top next : {out.top_tokens[:3]}")
        if out.activations is not None:
            print(f"Acts     : shape {out.activations.shape}")
        if out.attentions is not None:
            print(f"Attn     : shape {out.attentions.shape}")


def _cmd_probe(args: argparse.Namespace) -> None:
    """Handle the ``llmscope probe`` command (stub)."""
    print("[probe] Not yet implemented. Coming in Phase 3.", file=sys.stderr)
    sys.exit(0)


def _cmd_version(_args: argparse.Namespace) -> None:
    from llmscope import __version__

    print(f"llmscope {__version__}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmscope",
        description="Platform-agnostic LLM interpretability toolkit.",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- trace ----
    trace_p = sub.add_parser("trace", help="Trace a prompt through a provider.")
    trace_p.add_argument("prompt", help="The input text.")
    trace_p.add_argument(
        "--provider", "-p",
        default="mock",
        choices=["mock", "openai", "anthropic", "huggingface", "ollama"],
        help="Backend provider (default: mock).",
    )
    trace_p.add_argument("--model", "-m", default="", help="Model name/identifier.")
    trace_p.add_argument("--seed", type=int, default=42, help="RNG seed for mock provider.")
    trace_p.add_argument("--json", action="store_true", help="Output JSON.")
    trace_p.set_defaults(func=_cmd_trace)

    # ---- probe ----
    probe_p = sub.add_parser("probe", help="Run probes against cached activations.")
    probe_p.set_defaults(func=_cmd_probe)

    # ---- version ----
    version_p = sub.add_parser("version", help="Print version and exit.")
    version_p.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — called by ``llmscope`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
