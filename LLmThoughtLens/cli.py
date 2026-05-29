"""LLmThoughtLens CLI — typed command dispatch + entrypoint to the Textual TUI.

Commands
--------
* ``LLmThoughtLens tui``               launch the interactive Textual app
* ``LLmThoughtLens trace PROMPT``     run a trace and (optionally) save a report
* ``LLmThoughtLens probe``            run the full 10-probe battery and print scorecard
* ``LLmThoughtLens cache-activations`` collect HF activations for SAE training
* ``LLmThoughtLens train-sae``        train a TopK SAE on cached activations
* ``LLmThoughtLens label-features``   auto-label SAE features via an LLM
* ``LLmThoughtLens providers``        list available providers and their import status
* ``LLmThoughtLens version``          print the version string
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from LLmThoughtLens import __version__

_console = Console()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_provider(provider: str, model: str, api_key: str | None, base_url: str | None):
    from LLmThoughtLens.providers.registry import get_provider

    kwargs: dict[str, Any] = {}
    if provider == "openai":
        kwargs = {
            "model": model or "gpt-4o-mini",
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
        }
    elif provider == "anthropic":
        kwargs = {
            "model": model or "claude-3-5-haiku-20241022",
            "api_key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
        }
    elif provider == "huggingface":
        kwargs = {"model_name": model or "gpt2"}
    elif provider == "ollama":
        kwargs = {"model": model or "llama3.2", "base_url": base_url or "http://localhost:11434"}
    elif provider == "mock":
        pass
    return get_provider(provider, **kwargs)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_tui(_argv: list[str]) -> int:
    from LLmThoughtLens.tui.app import run_tui

    run_tui()
    return 0


def cmd_trace(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="LLmThoughtLens trace")
    p.add_argument("prompt", help="prompt text")
    p.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "openai", "anthropic", "huggingface", "ollama"],
    )
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--probes", action="store_true")
    p.add_argument("--output", default=None, help="write HTML report to this path")
    p.add_argument("--json", action="store_true", help="emit JSON summary on stdout")
    args = p.parse_args(argv)

    from LLmThoughtLens.scope import Scope

    provider = _make_provider(args.provider, args.model, args.api_key, args.base_url)
    scope = Scope(provider, top_k_features=args.top_k, attribution_threshold=args.threshold)
    result = scope.trace_full(args.prompt, run_probes=args.probes)

    if args.output:
        result.save(args.output)
        _console.print(f"wrote HTML report to {args.output}")

    if args.json:
        summary = {
            "prompt": result.prompt,
            "output_token": result.output_token,
            "top_tokens": result.top_tokens,
            "evidence_kind": result.evidence_kind,
            "n_features": len(result.features),
            "n_graph_nodes": result.graph.num_nodes,
            "n_graph_edges": result.graph.num_edges,
            "probes": [r.as_dict() for r in result.probe_results],
        }
        print(json.dumps(summary, indent=2, default=str))
    else:
        _console.print(f"[b]provider[/b]  {provider.model_id}")
        _console.print(f"[b]evidence[/b]  {result.evidence_kind}")
        _console.print(f"[b]output[/b]    {result.output_token}  ({result.output.output_prob:.2f})")
        _console.print(f"[b]features[/b]  {len(result.features)}")
        _console.print(
            f"[b]graph[/b]     {result.graph.num_nodes} nodes, {result.graph.num_edges} edges"
        )
        if result.probe_results:
            n_passed = sum(1 for r in result.probe_results if r.passed)
            _console.print(f"[b]probes[/b]    {n_passed} / {len(result.probe_results)} passed")
    return 0


def cmd_probe(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="LLmThoughtLens probe")
    p.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "openai", "anthropic", "huggingface", "ollama"],
    )
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--output", default=None, help="write JSON scorecard to this path")
    args = p.parse_args(argv)

    from LLmThoughtLens.probes.builtin import all_probes
    from LLmThoughtLens.probes.runner import ProbeRunner

    provider = _make_provider(args.provider, args.model, args.api_key, args.base_url)
    report = ProbeRunner(all_probes()).run_all(provider)

    tbl = Table(title=f"Probe scorecard — {provider.model_id}")
    tbl.add_column("Probe")
    tbl.add_column("Pass?", justify="center")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Summary")
    for r in report.results:
        tbl.add_row(
            r.probe_name,
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            f"{r.score:.2f}",
            r.summary[:80],
        )
    _console.print(tbl)
    _console.print(
        f"[b]overall[/b]  {report.n_passed} / {report.n_total} passed (mean {report.mean_score:.2f})"
    )

    if args.output:
        Path(args.output).write_text(report.to_json() or "")
        _console.print(f"wrote scorecard to {args.output}")
    return 0


def cmd_cache_activations(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="LLmThoughtLens cache-activations")
    p.add_argument("--provider", default="huggingface")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--corpus", required=True, help="path to a file with one prompt per line")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=10_000)
    p.add_argument("--output", required=True, help="output .npz path")
    args = p.parse_args(argv)

    from LLmThoughtLens.features.cache import ActivationCache

    provider = _make_provider(args.provider, args.model, None, None)
    cache = ActivationCache(provider, layer=args.layer, max_tokens=args.max_tokens)
    prompts = Path(args.corpus).read_text().splitlines()
    cache.collect(prompts, verbose=True)
    cache.save(args.output)
    _console.print(f"saved {len(cache)} tokens × {cache.d_model} to {args.output}")
    return 0


def cmd_train_sae(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="LLmThoughtLens train-sae")
    p.add_argument("--activations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dict-size", type=int, default=16384)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--l1", type=float, default=8e-4)
    args = p.parse_args(argv)

    from LLmThoughtLens.features.cache import ActivationCache
    from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

    data = ActivationCache.load(args.activations)
    acts = data["activations"]
    cfg = SAEConfig(
        input_dim=acts.shape[1],
        dict_size=args.dict_size,
        k=args.k,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_coeff=args.l1,
    )
    sae = SparseAutoencoder(cfg)
    _console.print(f"training SAE on {acts.shape[0]} × {acts.shape[1]} activations…")
    sae.fit(acts, verbose=True)
    sae.save(args.output)
    stats = sae.sparsity_stats(acts[: min(2048, len(acts))])
    _console.print(
        f"l0_mean={stats['l0_mean']:.1f}  dead={stats['dead_fraction']:.2%}  mse={stats['mse']:.4f}"
    )
    _console.print(f"saved SAE to {args.output}")
    return 0


def cmd_label_features(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="LLmThoughtLens label-features")
    p.add_argument("--sae", required=True)
    p.add_argument("--activations", required=True)
    p.add_argument("--corpus", required=True, help="corpus file (one prompt per line)")
    p.add_argument("--labeler-provider", default="openai")
    p.add_argument("--labeler-model", default="gpt-4o-mini")
    p.add_argument("--api-key", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    from LLmThoughtLens.features.cache import ActivationCache
    from LLmThoughtLens.features.labeler import FeatureLabeler
    from LLmThoughtLens.features.sae import SparseAutoencoder

    sae = SparseAutoencoder.load(args.sae)
    cache = ActivationCache.load(args.activations)
    acts = cache["activations"]
    streams = [line.split() for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    provider = _make_provider(args.labeler_provider, args.labeler_model, args.api_key, None)
    labeler = FeatureLabeler(sae, provider)
    labels = labeler.label_all(acts, streams, verbose=True)
    sae.save_with_labels(args.output, labels)
    _console.print(f"labelled {len(labels)} features → {args.output}")
    return 0


def cmd_providers(_argv: list[str]) -> int:
    from LLmThoughtLens.providers.registry import available_providers, list_providers

    tbl = Table(title="LLmThoughtLens providers")
    tbl.add_column("name")
    tbl.add_column("available?", justify="center")
    avail = set(available_providers())
    for name in list_providers():
        tbl.add_row(
            name, "[green]yes[/green]" if name in avail else "[yellow]missing extras[/yellow]"
        )
    _console.print(tbl)
    return 0


def cmd_version(_argv: list[str]) -> int:
    print(f"LLmThoughtLens {__version__}")
    return 0


def cmd_benchmark(argv: list[str]) -> int:
    """Run the full 10-probe battery, dump JSON, print a Rich scorecard."""
    import argparse

    p = argparse.ArgumentParser(
        prog="LLmThoughtLens benchmark",
        description=(
            "Run the 10 built-in interpretability probes against a provider, "
            "write a JSON scorecard, and print a formatted summary."
        ),
    )
    p.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "openai", "anthropic", "huggingface", "ollama"],
    )
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument(
        "--output",
        default="benchmark_results.json",
        help="path to write the JSON scorecard (default: benchmark_results.json)",
    )
    args = p.parse_args(argv)

    from LLmThoughtLens.probes.builtin import all_probes
    from LLmThoughtLens.probes.runner import ProbeRunner

    provider = _make_provider(args.provider, args.model, args.api_key, args.base_url)
    _console.print(f"[b]LLmThoughtLens benchmark[/b]  provider=[b]{provider.model_id}[/b]")
    report = ProbeRunner(all_probes()).run_all(provider)

    tbl = Table(title=f"Interpretability scorecard — {provider.model_id}")
    tbl.add_column("Probe")
    tbl.add_column("Pass?", justify="center")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Summary")
    for r in report.results:
        tbl.add_row(
            r.probe_name,
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            f"{r.score:.2f}",
            (r.summary or "")[:80],
        )
    _console.print(tbl)
    _console.print(
        f"[b]overall[/b]  {report.n_passed} / {report.n_total} passed "
        f"(mean {report.mean_score:.2f})"
    )

    out_text = report.to_json() or ""
    Path(args.output).write_text(out_text)
    _console.print(f"wrote {Path(args.output).resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_COMMANDS = {
    "tui": cmd_tui,
    "trace": cmd_trace,
    "probe": cmd_probe,
    "benchmark": cmd_benchmark,
    "cache-activations": cmd_cache_activations,
    "train-sae": cmd_train_sae,
    "label-features": cmd_label_features,
    "providers": cmd_providers,
    "version": cmd_version,
}


def _help() -> None:
    _console.print("[b]LLmThoughtLens[/b] — platform-agnostic LLM interpretability\n")
    _console.print("Usage: LLmThoughtLens <command> [options]\n")
    _console.print("Commands:")
    for name in _COMMANDS:
        _console.print(f"  {name}")
    _console.print("\nRun 'LLmThoughtLens <command> --help' for command-specific options.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _help()
        return 0
    cmd = args.pop(0)
    fn = _COMMANDS.get(cmd)
    if fn is None:
        _console.print(f"[red]unknown command:[/red] {cmd}")
        _help()
        return 2
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
