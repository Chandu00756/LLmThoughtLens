"""LLmThoughtLens CLI — typed command dispatch + entrypoint to the Textual TUI.

Commands
--------
* ``LLmThoughtLens tui``               launch the interactive Textual app
* ``LLmThoughtLens trace PROMPT``     run a trace and (optionally) save a report;
  ``--attribution`` / ``--validate`` / ``--metric`` / ``--scoring`` / ``--sae``
* ``LLmThoughtLens steer PROMPT``     activation steering: baseline vs steered completion
* ``LLmThoughtLens sae``              list pretrained SAE releases / inspect an SAE
* ``LLmThoughtLens probe``            run the full 10-probe battery and print scorecard
* ``LLmThoughtLens benchmark``        probe battery for one provider, or a model matrix
  (``--models hf:gpt2,mock --out-dir bench``) with JSON / Markdown / HTML scorecards
* ``LLmThoughtLens cache-activations`` collect HF activations for SAE training
* ``LLmThoughtLens train-sae``        train a TopK SAE on cached activations
* ``LLmThoughtLens label-features``   auto-label SAE features via an LLM
* ``LLmThoughtLens providers``        list available providers and their import status
* ``LLmThoughtLens version``          print the version string

Attribution defaults for ``trace`` match the other user-facing surfaces
(:data:`LLmThoughtLens.scope.ATTRIBUTION_DEFAULTS`): ``--attribution auto``
(gradient edges on HuggingFace models), ``--metric logprob`` and
``--attribution-nodes 10``; ``--validate K`` runs K real ablations and reports
Spearman, Pearson, n and sign agreement together.

Output streams: a command's *result* (trace summary, scorecard table, JSON)
goes to stdout; status chatter — files written, progress lines, warnings and
errors about inputs — goes to stderr, so ``LLmThoughtLens trace … --json |
jq`` always receives clean JSON.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from LLmThoughtLens import __version__

#: Results (summaries, tables) — stdout.
_console = Console()
#: Status chatter (files written, progress, warnings) — stderr.
_err_console = Console(stderr=True)

_PROVIDER_CHOICES = ["mock", "openai", "anthropic", "huggingface", "ollama"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_provider(provider: str, model: str, api_key: str | None, base_url: str | None):
    """Instantiate *provider*; empty model/url fall back to the shared defaults.

    Defaults live in :mod:`LLmThoughtLens.providers.defaults` (overridable via
    ``LLMTHOUGHTLENS_<PROVIDER>_MODEL``).  Unset API keys fall back to the
    provider's own environment variable (``OPENAI_API_KEY`` etc.).
    """
    from LLmThoughtLens.providers.defaults import provider_kwargs
    from LLmThoughtLens.providers.registry import get_provider

    kwargs = provider_kwargs(provider, model, api_key=api_key, base_url=base_url)
    return get_provider(provider, **kwargs)


def _error(message: str) -> int:
    """Print an input / capability error to stderr and return exit code 2."""
    _err_console.print(f"[red]error:[/red] {message}", markup=True, highlight=False)
    return 2


def _parse_sae_spec(spec: str) -> tuple[str, str | None]:
    """``"release:sae_id"`` -> ``(release, sae_id)``; an existing path -> ``(path, None)``."""
    if Path(spec).expanduser().exists():
        return spec, None
    if ":" in spec:
        release, sae_id = spec.split(":", 1)
        if release and sae_id:
            return release, sae_id
    raise ValueError(
        f"--sae {spec!r}: expected RELEASE:SAE_ID (e.g. gpt2-small-res-jb:blocks.6.hook_resid_pre; "
        "see `LLmThoughtLens sae list`) or the path of a saved SAE"
    )


def _load_sae(spec: str, *, local_files_only: bool = False) -> Any:
    from LLmThoughtLens.scope import _load_sae_spec

    release, sae_id = _parse_sae_spec(spec)
    return _load_sae_spec(release, sae_id, local_files_only=local_files_only)


def _attribution_lines(summary: dict[str, Any]) -> list[str]:
    """Human-readable attribution lines (edges, fallback, faithfulness + caveat)."""
    from LLmThoughtLens.visualization.attribution_view import format_faithfulness

    lines: list[str] = []
    label = summary.get("semantics_label") or "unknown"
    detail = [str(summary.get("method") or "?")]
    if summary.get("metric"):
        target = summary.get("target_token")
        detail.append(
            f"metric {summary['metric']}" + (f" of {target!r}" if target is not None else "")
        )
    lines.append(f"[b]edges[/b]     {label} ({', '.join(detail)})")
    if summary.get("method_fallback"):
        lines.append(f"[b]fallback[/b]  {summary['method_fallback']}")
    frac = summary.get("unexplained_fraction")
    if isinstance(frac, (int, float)):
        what = (
            "attribution mass" if summary.get("error_kind") == "attribution_mass" else "energy"
        )
        lines.append(f"[b]uncovered[/b] {100.0 * float(frac):.1f}% of {what}")
    faith = summary.get("faithfulness")
    if faith:
        lines.append(f"[b]faithful[/b]  {format_faithfulness(faith)}")
        lines.append(f"          {faith.get('caveat', '')}")
    elif summary.get("faithfulness_skipped"):
        lines.append(f"[b]faithful[/b]  not computed: {summary['faithfulness_skipped']}")
    return lines


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_tui(_argv: list[str]) -> int:
    from LLmThoughtLens.tui.app import run_tui

    run_tui()
    return 0


def cmd_trace(argv: list[str]) -> int:
    import argparse

    from LLmThoughtLens.scope import ATTRIBUTION_DEFAULTS

    p = argparse.ArgumentParser(prog="LLmThoughtLens trace")
    p.add_argument("prompt", help="prompt text")
    p.add_argument("--provider", default="mock", choices=_PROVIDER_CHOICES)
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--probes", action="store_true")
    p.add_argument("--output", default=None, help="write HTML report to this path")
    p.add_argument("--json", action="store_true", help="emit JSON summary on stdout")
    p.add_argument(
        "--scoring",
        default="centered",
        choices=["centered", "l2"],
        help="white-box feature score: unitless centred distance (default) or legacy raw L2 norm",
    )
    p.add_argument(
        "--attribution",
        default=ATTRIBUTION_DEFAULTS["attribution"],
        choices=["auto", "gradient", "activation_flow"],
        help=(
            "edge method: auto = gradient (causal, linearised estimate) when the provider "
            "exposes a differentiable model (huggingface), else activation flow "
            "(correlational); black-box providers always use input masking"
        ),
    )
    p.add_argument(
        "--validate",
        type=int,
        default=0,
        metavar="K",
        help=(
            "ablate the top-K attributed nodes for real and report Spearman, Pearson, n and "
            "sign agreement between predicted and measured effects (gradient traces only)"
        ),
    )
    p.add_argument(
        "--metric",
        default=ATTRIBUTION_DEFAULTS["metric"],
        choices=["logprob", "logit", "logit_diff"],
        help="target metric of gradient attribution (default logprob)",
    )
    p.add_argument(
        "--attribution-nodes",
        type=int,
        default=ATTRIBUTION_DEFAULTS["attribution_nodes"],
        metavar="N",
        help="add the N residual nodes with the largest attribution to the graph (gradient)",
    )
    p.add_argument(
        "--target",
        default=None,
        help="target token for gradient attribution (default: the model's top-1 prediction)",
    )
    p.add_argument(
        "--sae",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "attach an SAE: RELEASE:SAE_ID (e.g. gpt2-small-res-jb:blocks.6.hook_resid_pre) "
            "or a saved SAE path; repeat for several"
        ),
    )
    p.add_argument(
        "--sae-layer",
        type=int,
        default=None,
        help="hook layer for an SAE file without hook metadata",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="load --sae only from the local Hugging Face cache (no download)",
    )
    args = p.parse_args(argv)

    from LLmThoughtLens.scope import Scope

    if args.validate < 0 or args.attribution_nodes < 0:
        return _error("--validate and --attribution-nodes must be >= 0")
    provider = _make_provider(args.provider, args.model, args.api_key, args.base_url)
    scope = Scope(
        provider,
        top_k_features=args.top_k,
        attribution_threshold=args.threshold,
        scoring=args.scoring,
        attribution=args.attribution,
        metric=args.metric,
        attribution_nodes=args.attribution_nodes,
        validate=args.validate,
    )
    try:
        for i, spec in enumerate(args.sae):
            sae = _load_sae(spec, local_files_only=args.local_files_only)
            layer = args.sae_layer if args.sae_layer is not None and i == 0 else None
            if i == 0:
                scope.attach_sae(sae, layer)
            else:
                scope.add_sae(sae)
        result = scope.trace_full(args.prompt, run_probes=args.probes, target=args.target)
    except (ValueError, FileNotFoundError, KeyError, NotImplementedError) as exc:
        return _error(str(exc))

    if args.output:
        result.save(args.output)
        _err_console.print(f"wrote HTML report to {args.output}")
    for warning in result.meta.get("sae_warnings", []):
        _err_console.print(f"[yellow]warning:[/yellow] {warning}")

    summary_attr = result.attribution_summary()
    if args.json:
        input_tokens = result.input_tokens
        summary = {
            "prompt": result.prompt,
            "output_token": result.output_token,
            "top_tokens": result.top_tokens,
            "evidence_kind": result.evidence_kind,
            "n_features": len(result.features),
            "n_graph_nodes": result.graph.num_nodes,
            "n_graph_edges": result.graph.num_edges,
            "probes": [r.as_dict() for r in result.probe_results],
            "attribution": summary_attr,
            "saes": result.meta.get("saes", []),
            "sae_warnings": result.meta.get("sae_warnings", []),
            "top_features": [
                {**f.as_dict(), "token": _token_at(input_tokens, f.token_idx)}
                for f in result.top_features(10)
            ],
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
        for line in _attribution_lines(summary_attr):
            _console.print(line, highlight=False)
        if result.probe_results:
            n_passed = sum(1 for r in result.probe_results if r.passed)
            _console.print(f"[b]probes[/b]    {n_passed} / {len(result.probe_results)} passed")
    return 0


def _token_at(tokens: list[str], idx: int) -> str:
    return str(tokens[idx]) if 0 <= idx < len(tokens) else ""


def cmd_steer(argv: list[str]) -> int:
    """Activation steering: baseline vs steered completion on a local white-box model."""
    import argparse

    p = argparse.ArgumentParser(
        prog="LLmThoughtLens steer",
        description=(
            "Add a steering vector to a local model's residual stream and compare the steered "
            "completion with the baseline. The vector comes from contrast prompts "
            "(--positive/--negative, mean activation difference), a saved vector (--vector) or "
            "an SAE feature (--sae + --feature). Needs the huggingface provider: API models "
            "expose no residual stream and the mock provider's activations are synthetic."
        ),
    )
    p.add_argument("prompt", help="prompt to complete with and without steering")
    p.add_argument("--provider", default="huggingface", choices=_PROVIDER_CHOICES)
    p.add_argument("--model", default="", help="model id (default: the provider's default)")
    p.add_argument("--device", default="auto", help="huggingface device (auto/cpu/cuda/mps)")
    p.add_argument("--positive", action="append", default=[], help="positive prompt (repeat)")
    p.add_argument("--negative", action="append", default=[], help="negative prompt (repeat)")
    p.add_argument("--vector", default=None, help="saved steering vector (.npz)")
    p.add_argument("--sae", default=None, metavar="SPEC", help="SAE (RELEASE:SAE_ID or path)")
    p.add_argument("--feature", type=int, default=None, help="SAE feature id for --sae")
    p.add_argument("--layer", type=int, default=None, help="layer (required for contrast)")
    p.add_argument("--site", default="resid_post", choices=["resid_post", "resid_pre"])
    p.add_argument(
        "--position",
        default="last",
        choices=["last", "mean"],
        help="contrast vectors: read each prompt's last token or average its tokens",
    )
    p.add_argument(
        "--positions",
        default=None,
        choices=["all", "last", "prompt", "generated"],
        help="which token positions are steered (default all, or the saved vector's)",
    )
    p.add_argument("--coeff", type=float, default=None, help="coefficient (default: vector's)")
    p.add_argument("--normalize", action="store_true", help="steer along the unit direction")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
    p.add_argument("--save-vector", default=None, help="also save the vector (.npz)")
    p.add_argument(
        "--output",
        default="steering_result.json",
        help="write the full result JSON here (default: steering_result.json)",
    )
    p.add_argument("--report", default=None, help="also write an HTML steering report")
    p.add_argument("--json", action="store_true", help="print the result JSON on stdout")
    p.add_argument("--local-files-only", action="store_true", help="--sae from the cache only")
    args = p.parse_args(argv)

    from LLmThoughtLens.features.steering import (
        SteeringMismatchError,
        SteeringUnavailableError,
        SteeringVector,
    )
    from LLmThoughtLens.scope import Scope

    sources = [
        bool(args.positive or args.negative),
        args.vector is not None,
        args.sae is not None,
    ]
    if sum(sources) != 1:
        return _error(
            "choose exactly one vector source: --positive/--negative, --vector, or --sae --feature"
        )
    if (args.positive or args.negative) and not (args.positive and args.negative):
        return _error("contrast steering needs at least one --positive and one --negative prompt")
    if args.max_new_tokens < 1:
        return _error("--max-new-tokens must be >= 1")

    kwargs: dict[str, Any] = {"device": args.device} if args.provider == "huggingface" else {}
    try:
        provider = _make_provider_kw(args.provider, args.model, **kwargs)
        scope = Scope(provider)
        hooked = scope.hooked
        if args.vector is not None:
            vector = SteeringVector.load(args.vector, hooked=hooked)
            if args.layer is not None or args.site != "resid_post" or args.positions:
                vector = SteeringVector.from_vector(
                    vector.direction,
                    args.layer if args.layer is not None else vector.layer,
                    site=args.site,  # type: ignore[arg-type]
                    coeff=vector.coeff,
                    normalize=vector.normalize,
                    positions=args.positions or vector.positions,
                    name=vector.name,
                    model_name=vector.model_name,
                    source=vector.source,
                )
        elif args.sae is not None:
            if args.feature is None:
                return _error("--sae needs --feature FID")
            sae = _load_sae(args.sae, local_files_only=args.local_files_only)
            vector = SteeringVector.from_sae_feature(
                sae,
                args.feature,
                layer=args.layer,
                coeff=1.0,
                positions=args.positions or "all",  # type: ignore[arg-type]
            )
        else:
            if args.layer is None:
                return _error("contrast steering needs --layer")
            vector = scope.steering_vector_from_contrast(
                args.positive,
                args.negative,
                args.layer,
                args.site,
                args.position,
                normalize=args.normalize,
                positions=args.positions or "all",
                chat=args.chat,
            )
        if args.normalize and not vector.normalize:
            vector = SteeringVector.from_vector(
                vector.direction,
                vector.layer,
                site=vector.site,
                coeff=vector.coeff,
                normalize=True,
                positions=vector.positions,
                name=vector.name,
                model_name=vector.model_name,
                source=vector.source,
            )
        if args.coeff is not None:
            vector = vector.with_coeff(args.coeff)
        result = scope.steer(
            args.prompt,
            vector,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            chat=args.chat,
        )
    except SteeringUnavailableError as exc:
        return _error(f"{exc} Try: --provider huggingface --model gpt2")
    except (SteeringMismatchError, ValueError, FileNotFoundError, KeyError) as exc:
        return _error(str(exc))

    data = result.to_dict()
    data["model"] = provider.model_id
    if args.save_vector:
        path = vector.save(args.save_vector)
        _err_console.print(f"saved steering vector to {path}")
    if args.output:
        Path(args.output).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        _err_console.print(f"wrote steering result to {args.output}")
    if args.report:
        from LLmThoughtLens.visualization.report import ReportBuilder

        ReportBuilder.from_steering_result(data).save(args.report)
        _err_console.print(f"wrote HTML report to {args.report}")

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0
    v = data["vectors"][0] if data.get("vectors") else {}
    diverged = data.get("diverged_at")
    _console.print(
        f"[b]model[/b]      {provider.model_id}  layer {v.get('layer')} {v.get('site')}  "
        f"coeff {v.get('coeff')}  positions {v.get('positions')}",
        highlight=False,
    )
    _console.print(f"[b]prompt[/b]     {data['prompt']!r}", highlight=False)
    _console.print(f"[b]baseline[/b]   {data['baseline']['text']!r}", highlight=False)
    _console.print(f"[b]steered[/b]    {data['steered']['text']!r}", highlight=False)
    _console.print(
        f"[b]KL[/b]         first step {data['first_step_kl']:.4f} nats, mean "
        f"{data['mean_kl']:.4f} over {len(data['kl_per_step'])} steps; "
        + ("identical completions" if diverged is None else f"diverged at token {diverged}"),
        highlight=False,
    )
    for label, rows in (("promoted", data["promoted"]), ("suppressed", data["suppressed"])):
        shown = ", ".join(
            f"{r['token']!r} {r['p_baseline']:.3f}->{r['p_steered']:.3f}" for r in rows[:5]
        )
        _console.print(f"[b]{label:<10}[/b] {shown or '-'}", highlight=False)
    _console.print(
        f"[b]evidence[/b]   {data['evidence_kind']} · {data['method']} · "
        f"{data['effect_semantics']}",
        highlight=False,
    )
    _console.print(f"           {data['note']}", highlight=False)
    return 0


def _make_provider_kw(provider: str, model: str, **extra: Any) -> Any:
    """:func:`_make_provider` plus extra provider kwargs (e.g. a HuggingFace device)."""
    from LLmThoughtLens.providers.defaults import provider_kwargs
    from LLmThoughtLens.providers.registry import get_provider

    kwargs = provider_kwargs(provider, model, api_key=None, base_url=None)
    kwargs.update(extra)
    return get_provider(provider, **kwargs)


def cmd_sae(argv: list[str]) -> int:
    """List pretrained SAE releases, or load and inspect one SAE."""
    import argparse

    p = argparse.ArgumentParser(
        prog="LLmThoughtLens sae",
        description=(
            "Pretrained sparse autoencoders. `list` shows the known releases (no network); "
            "`inspect SPEC` loads one (RELEASE:SAE_ID from the Hugging Face cache or Hub, or a "
            "saved SAE path) and prints its hook point and configuration. Attach SAEs to a "
            "trace with `LLmThoughtLens trace PROMPT --provider huggingface --sae SPEC`."
        ),
    )
    sub = p.add_subparsers(dest="action")
    p_list = sub.add_parser("list", help="known pretrained SAE releases")
    p_list.add_argument("--json", action="store_true")
    p_inspect = sub.add_parser("inspect", help="load one SAE and print its configuration")
    p_inspect.add_argument("spec", help="RELEASE:SAE_ID or a saved SAE path")
    p_inspect.add_argument("--local-files-only", action="store_true", help="cache only")
    p_inspect.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if args.action is None:
        p.print_help()
        return 0
    if args.action == "list":
        from LLmThoughtLens.features.sae_loaders import list_pretrained

        releases = list_pretrained()
        if args.json:
            print(json.dumps(releases, indent=2))
            return 0
        tbl = Table(title="Pretrained SAE releases")
        for col in ("release", "model", "format", "example sae_id", "description"):
            tbl.add_column(col)
        for name, info in releases.items():
            tbl.add_row(
                name, info["model"], info["format"], info["example_sae_id"], info["description"]
            )
        _console.print(tbl)
        _console.print("Any Hub repo id also works as a release (org/repo:folder).")
        return 0

    try:
        sae = _load_sae(args.spec, local_files_only=args.local_files_only)
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        return _error(str(exc))
    cfg = sae.config
    info = {
        "spec": args.spec,
        "hook_name": cfg.hook_name,
        "hook_layer": cfg.hook_layer,
        "hook_site": cfg.hook_site,
        "d_in": cfg.input_dim,
        "d_sae": cfg.dict_size,
        "architecture": cfg.architecture,
        "k": cfg.k if cfg.architecture == "topk" else None,
        "normalize_activations": cfg.normalize_activations,
        "apply_b_dec_to_input": cfg.apply_b_dec_to_input,
        "center_input": cfg.center_input,
        "model_name": cfg.model_name,
        "hf_model": (cfg.extra or {}).get("hf_model"),
        "release": cfg.release,
        "sae_id": cfg.sae_id,
        "source_format": cfg.source_format,
        "prepend_bos": (cfg.extra or {}).get("prepend_bos"),
        "n_labels": len(sae.labels),
    }
    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return 0
    tbl = Table(title=f"SAE {args.spec}")
    tbl.add_column("field")
    tbl.add_column("value")
    for key, value in info.items():
        tbl.add_row(key, str(value))
    _console.print(tbl)
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
        _err_console.print(f"wrote scorecard to {args.output}")
    return 0


def cmd_cache_activations(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="LLmThoughtLens cache-activations")
    p.add_argument("--provider", default="huggingface")
    p.add_argument("--model", default="", help="model id (default: the provider's default)")
    p.add_argument(
        "--corpus",
        required=True,
        help="path to a file with one prompt per line (blank lines are skipped)",
    )
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=10_000)
    p.add_argument("--output", required=True, help="output .npz path")
    args = p.parse_args(argv)

    from LLmThoughtLens.features.cache import ActivationCache

    provider = _make_provider(args.provider, args.model, None, None)
    cache = ActivationCache(provider, layer=args.layer, max_tokens=args.max_tokens)
    # Blank lines are skipped (they carry no tokens to label); the 1-based
    # corpus line of every prompt is stored with the provider's own tokens.
    numbered = [
        (n, line)
        for n, line in enumerate(Path(args.corpus).read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    ]
    with contextlib.redirect_stdout(sys.stderr):  # per-prompt progress is chatter
        cache.collect(
            [line for _, line in numbered],
            verbose=True,
            line_numbers=[n for n, _ in numbered],
        )
    cache.save(args.output)
    _err_console.print(f"saved {len(cache)} tokens × {cache.d_model} to {args.output}")
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
    _err_console.print(f"training SAE on {acts.shape[0]} × {acts.shape[1]} activations…")
    with contextlib.redirect_stdout(sys.stderr):  # training progress is chatter
        sae.fit(acts, verbose=True)
    sae.save(args.output)
    stats = sae.sparsity_stats(acts[: min(2048, len(acts))])
    _console.print(
        f"l0_mean={stats['l0_mean']:.1f}  dead={stats['dead_fraction']:.2%}  mse={stats['mse']:.4f}"
    )
    _err_console.print(f"saved SAE to {args.output}")
    return 0


def cmd_label_features(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="LLmThoughtLens label-features")
    p.add_argument("--sae", required=True)
    p.add_argument("--activations", required=True, help=".npz written by cache-activations")
    p.add_argument(
        "--corpus",
        default=None,
        help=(
            "corpus file (one prompt per line); only needed for activation caches written "
            "before token storage — current caches carry the provider's own tokens"
        ),
    )
    p.add_argument("--labeler-provider", default="openai")
    p.add_argument(
        "--labeler-model", default="", help="model id (default: the labeler provider's default)"
    )
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--output",
        required=True,
        help=(
            "where to write the labelled SAE (e.g. sae_labelled.pt): the --sae weights and "
            "config with the new feature labels merged in, in the same SAE file format "
            "train-sae writes (load it with SparseAutoencoder.load). It is not an "
            "activations .npz"
        ),
    )
    args = p.parse_args(argv)

    from LLmThoughtLens.features.cache import ActivationCache
    from LLmThoughtLens.features.labeler import FeatureLabeler
    from LLmThoughtLens.features.sae import SparseAutoencoder

    cache = ActivationCache.load(args.activations)
    acts = cache["activations"]
    aligned = _label_token_streams(cache, args.activations, args.corpus)
    if aligned is None:
        return 2
    streams, positions = aligned
    sae = SparseAutoencoder.load(args.sae)
    provider = _make_provider(args.labeler_provider, args.labeler_model, args.api_key, None)
    labeler = FeatureLabeler(sae, provider)
    with contextlib.redirect_stdout(sys.stderr):  # per-feature progress is chatter
        labels = labeler.label_all(acts, streams, verbose=True, positions=positions)
    sae.save_with_labels(args.output, labels)
    _err_console.print(f"labelled {len(labels)} features → {args.output}")
    return 0


def _label_token_streams(
    cache: dict[str, Any], activations_path: str, corpus: str | None
) -> tuple[list[list[str]], list[tuple[int, int]] | None] | None:
    """Token contexts aligned 1:1 with the cached activation rows.

    Current caches store the provider's own tokens, which are used directly
    (*corpus* is ignored).  For caches written before token storage the corpus
    is whitespace-split — accepted only when the token count matches the
    activation rows, and with a warning, because the provider's tokenizer
    (e.g. GPT-2 BPE) may split text differently.  Returns ``None`` after
    printing an error when no safe alignment exists.
    """
    from LLmThoughtLens.features.cache import ActivationCache

    n_rows = int(cache["activations"].shape[0])
    if ActivationCache.has_tokens(cache):
        try:
            streams, positions = ActivationCache.token_streams(cache)
        except ValueError as exc:
            _err_console.print(f"[red]error:[/red] {activations_path}: {exc}")
            return None
        if corpus:
            _err_console.print(
                f"note: using the provider tokens stored in {activations_path}; --corpus is ignored"
            )
        if not cache["meta"].get("tokens_aligned", True):
            _err_console.print(
                "[yellow]warning:[/yellow] the provider's tokens did not match its activation "
                "rows when this cache was written; some contexts may be misaligned"
            )
        return streams, positions

    provider_name = cache["meta"].get("provider", "the provider")
    if not corpus:
        _err_console.print(
            f"[red]error:[/red] {activations_path} predates token storage, so its rows can't "
            "be mapped back to text. Re-run `LLmThoughtLens cache-activations` (it now stores "
            "the provider's tokens), or pass --corpus with the exact corpus used."
        )
        return None
    streams = [
        line.split()
        for line in Path(corpus).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_words = sum(len(s) for s in streams)
    if n_words != n_rows:
        _err_console.print(
            f"[red]error:[/red] --corpus has {n_words} whitespace tokens but {activations_path} "
            f"has {n_rows} activation rows ({provider_name} tokenizes differently), so feature "
            "contexts can't be aligned. Re-run `LLmThoughtLens cache-activations` — it now "
            "stores the provider's own tokens — and label again."
        )
        return None
    _err_console.print(
        f"[yellow]warning:[/yellow] {activations_path} predates token storage; assuming "
        f"{provider_name} tokenized --corpus on whitespace. Alignment is not guaranteed — "
        "re-run `LLmThoughtLens cache-activations` to store the provider's tokens."
    )
    return streams, None


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


def cmd_serve(argv: list[str]) -> int:
    """Launch the live web dashboard + provider-compatible proxy."""
    import argparse

    p = argparse.ArgumentParser(
        prog="LLmThoughtLens serve",
        description=(
            "Start the live interpretability dashboard. Open the printed URL in a "
            "browser to configure providers, trace prompts live, and route apps "
            "through the OpenAI-compatible proxy at <url>/v1."
        ),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    args = p.parse_args(argv)

    try:
        from LLmThoughtLens.server.app import run_server
    except ImportError:
        _console.print(
            "[red]The dashboard needs the 'server' extra.[/red]\n"
            "Install with: [b]pip install 'LLmThoughtLens[server]'[/b]"
        )
        return 1

    url = f"http://{args.host}:{args.port}/"
    _console.print(f"[b]LLmThoughtLens Live[/b] → {url}")
    _console.print(f"OpenAI-compatible proxy → [b]{url}v1[/b]  (point any app's base URL here)")
    run_server(host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


def cmd_benchmark(argv: list[str]) -> int:
    """Probe benchmark: one provider (legacy JSON), or a model matrix with scorecards."""
    import argparse

    p = argparse.ArgumentParser(
        prog="LLmThoughtLens benchmark",
        description=(
            "Run the 10 built-in interpretability probes. Without --models / --out-dir: one "
            "provider, a JSON scorecard (--output) and a printed summary. With --models "
            "(e.g. --models hf:gpt2,mock --models ollama:qwen3:1.7b) or --out-dir: the matrix "
            "runner (repeats, seeds, failure isolation, timing, cost) writing a versioned JSON "
            "record plus Markdown and HTML scorecards to --out-dir."
        ),
    )
    p.add_argument("--provider", default="mock", choices=_PROVIDER_CHOICES)
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument(
        "--output",
        default=None,
        help=(
            "JSON path. Single-provider mode: the scorecard (default: benchmark_results.json). "
            "Matrix mode: an extra copy of the JSON record."
        ),
    )
    p.add_argument(
        "--models",
        action="append",
        default=[],
        metavar="SPECS",
        help=(
            "matrix mode: model specs, comma-separated and/or repeated "
            "(mock, hf:gpt2, ollama:qwen3:1.7b, openai:<model>, anthropic:<model>)"
        ),
    )
    p.add_argument("--out-dir", default=None, help="matrix mode: scorecard directory")
    p.add_argument("--repeats", type=int, default=1, help="matrix mode: repeats per cell")
    p.add_argument(
        "--seeds", default=None, help="matrix mode: comma-separated seeds (overrides --repeats)"
    )
    p.add_argument("--temperature", type=float, default=0.0, help="matrix mode: temperature")
    p.add_argument("--probes", default=None, help="matrix mode: comma-separated probe names")
    p.add_argument(
        "--chart",
        default="inline",
        choices=["inline", "cdn", "svg", "none"],
        help="matrix mode: how the HTML scorecard embeds its chart",
    )
    p.add_argument("--stem", default="scorecard", help="matrix mode: output file stem")
    args = p.parse_args(argv)

    if args.models or args.out_dir:
        return _benchmark_matrix(args)

    from LLmThoughtLens.probes.builtin import all_probes
    from LLmThoughtLens.probes.runner import ProbeRunner

    output = args.output or "benchmark_results.json"
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
    Path(output).write_text(out_text)
    _err_console.print(f"wrote {Path(output).resolve()}")
    return 0


def _benchmark_matrix(args: Any) -> int:
    """``benchmark --models ... --out-dir ...`` through :func:`LLmThoughtLens.bench.run_and_write`."""
    from LLmThoughtLens.bench import run_and_write

    specs = [s.strip() for chunk in args.models for s in chunk.split(",") if s.strip()]
    if not specs:
        model = args.model or ""
        specs = [args.provider if args.provider == "mock" else f"{args.provider}:{model}"]
        specs = [s.rstrip(":") for s in specs]
    seeds = None
    if args.seeds:
        try:
            seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        except ValueError:
            return _error(f"--seeds {args.seeds!r}: expected comma-separated integers")
    probes = [x.strip() for x in args.probes.split(",") if x.strip()] if args.probes else None
    out_dir = args.out_dir or "bench-out"

    def progress(event: dict[str, Any]) -> None:
        kind = event.get("event") or event.get("kind") or event.get("type")
        if kind == "cell_done":
            score = event.get("score")
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "-"
            _err_console.print(
                f"[{event.get('index', '?')}/{event.get('total', '?')}] "
                f"{event.get('model', '')} · {event.get('probe', '')}: "
                f"{event.get('status', '')} score={score_text}",
                highlight=False,
                markup=False,
            )
        elif kind in ("model_start", "model_error"):
            detail = f" ({event.get('error')})" if event.get("error") else ""
            _err_console.print(
                f"{kind}: {event.get('model', '')}{detail}", highlight=False, markup=False
            )

    _console.print(f"[b]LLmThoughtLens benchmark[/b]  models=[b]{', '.join(specs)}[/b]")
    try:
        result, paths = run_and_write(
            specs,
            out_dir,
            probes,
            stem=args.stem,
            plotlyjs=args.chart,
            progress=progress,
            repeats=args.repeats,
            seeds=seeds,
            temperature=args.temperature,
        )
    except (ValueError, KeyError) as exc:
        return _error(str(exc))

    tbl = Table(title="Benchmark scorecard (per model)")
    for col in ("model", "status", "passed", "mean score", "error cells", "synthetic"):
        tbl.add_column(col)
    for row in result.aggregates.get("by_model", []):
        mean = row.get("mean_score")
        tbl.add_row(
            str(row.get("model")),
            str(row.get("status")),
            f"{row.get('n_passed')} / {row.get('n_scored')}",
            f"{mean:.2f}" if isinstance(mean, (int, float)) else "-",
            str(row.get("n_error_cells")),
            "yes" if row.get("synthetic") else "no",
        )
    _console.print(tbl)
    if args.output:
        result.to_json(args.output)
        _err_console.print(f"wrote {Path(args.output).resolve()}")
    for fmt, path in paths.items():
        _err_console.print(f"wrote {fmt} scorecard to {Path(path).resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_COMMANDS = {
    "serve": cmd_serve,
    "tui": cmd_tui,
    "trace": cmd_trace,
    "steer": cmd_steer,
    "sae": cmd_sae,
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
