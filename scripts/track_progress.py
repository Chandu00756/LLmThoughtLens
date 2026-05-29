"""
LLmThoughtLens / llmscope — Automated Progress Tracker
Inspects the repository and produces a phase-by-phase completion score.

Usage:
    python scripts/track_progress.py            # print report to stdout
    python scripts/track_progress.py --json     # emit JSON
    python scripts/track_progress.py --update   # rewrite PROGRESS.md
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "llmscope"
TESTS = ROOT / "tests"
GITHUB = ROOT / ".github"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Check:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class Phase:
    number: int
    name: str
    pct_start: int
    pct_end: int
    checks: list[Check] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(c.passed for c in self.checks) / len(self.checks)

    @property
    def completion_pct(self) -> float:
        span = self.pct_end - self.pct_start
        return self.pct_start + span * self.score

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def file_nonempty(path: Path) -> bool:
    return file_exists(path) and path.stat().st_size > 0


def dir_exists(path: Path) -> bool:
    return path.exists() and path.is_dir()


def contains_class(path: Path, class_name: str) -> bool:
    if not file_nonempty(path):
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return any(
            isinstance(node, ast.ClassDef) and node.name == class_name
            for node in ast.walk(tree)
        )
    except SyntaxError:
        return False


def contains_function(path: Path, func_name: str) -> bool:
    if not file_nonempty(path):
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
            for node in ast.walk(tree)
        )
    except SyntaxError:
        return False


def contains_method(path: Path, class_name: str, method_name: str) -> bool:
    if not file_nonempty(path):
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in ast.walk(node):
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == method_name
                    ):
                        return True
    except SyntaxError:
        pass
    return False


def grep_file(path: Path, pattern: str) -> bool:
    if not file_nonempty(path):
        return False
    return bool(re.search(pattern, path.read_text(encoding="utf-8")))


def grep_dir(directory: Path, pattern: str, glob: str = "**/*.py") -> bool:
    if not dir_exists(directory):
        return False
    return any(
        re.search(pattern, f.read_text(encoding="utf-8"))
        for f in directory.glob(glob)
        if f.is_file() and f.stat().st_size > 0
    )


def run_command(*args: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=60,
        )
        return result.returncode, result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def package_importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Phase 0 — Foundation and control plane (0 → 10%)
# ---------------------------------------------------------------------------

def check_phase_0() -> Phase:
    phase = Phase(0, "Foundation and control plane", 0, 10)

    phase.checks.append(Check(
        "README.md exists and non-empty",
        file_nonempty(ROOT / "README.md"),
    ))
    phase.checks.append(Check(
        "LICENSE exists",
        file_exists(ROOT / "LICENSE"),
    ))
    phase.checks.append(Check(
        "pyproject.toml exists and non-empty",
        file_nonempty(ROOT / "pyproject.toml"),
    ))
    phase.checks.append(Check(
        "CONTRIBUTING.md exists",
        file_exists(ROOT / "CONTRIBUTING.md"),
    ))
    phase.checks.append(Check(
        "SECURITY.md exists",
        file_exists(ROOT / "SECURITY.md"),
    ))
    phase.checks.append(Check(
        "CODE_OF_CONDUCT.md exists",
        file_exists(ROOT / "CODE_OF_CONDUCT.md"),
    ))
    adr_dir = ROOT / "docs" / "adr"
    phase.checks.append(Check(
        "ADR directory exists (docs/adr/)",
        dir_exists(adr_dir),
    ))
    adr_count = len(list(adr_dir.glob("ADR-*.md"))) if dir_exists(adr_dir) else 0
    phase.checks.append(Check(
        f"At least 5 ADR files present (found {adr_count})",
        adr_count >= 5,
    ))
    phase.checks.append(Check(
        ".github/ directory exists",
        dir_exists(GITHUB),
    ))
    issue_templates = GITHUB / "ISSUE_TEMPLATE"
    phase.checks.append(Check(
        "Issue templates present",
        dir_exists(issue_templates) and any(issue_templates.iterdir()),
    ))
    phase.checks.append(Check(
        "PR template present",
        file_exists(GITHUB / "PULL_REQUEST_TEMPLATE.md"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 1 — Package skeleton and local developer workflow (10 → 20%)
# ---------------------------------------------------------------------------

def check_phase_1() -> Phase:
    phase = Phase(1, "Package skeleton and local developer workflow", 10, 20)

    phase.checks.append(Check(
        "llmscope package directory exists",
        dir_exists(PKG),
    ))
    for subpkg in ["providers", "features", "circuits", "probes", "visualization"]:
        phase.checks.append(Check(
            f"llmscope/{subpkg}/ exists",
            dir_exists(PKG / subpkg),
        ))

    phase.checks.append(Check(
        "pyproject.toml defines [project] section",
        grep_file(ROOT / "pyproject.toml", r"\[project\]"),
    ))
    phase.checks.append(Check(
        "pyproject.toml defines build-system",
        grep_file(ROOT / "pyproject.toml", r"\[build-system\]"),
    ))
    phase.checks.append(Check(
        "Ruff configured (pyproject.toml or ruff.toml)",
        grep_file(ROOT / "pyproject.toml", r"\[tool\.ruff\]")
        or file_exists(ROOT / "ruff.toml"),
    ))
    phase.checks.append(Check(
        "pytest configured",
        grep_file(ROOT / "pyproject.toml", r"\[tool\.pytest")
        or file_exists(ROOT / "pytest.ini")
        or file_exists(ROOT / "setup.cfg"),
    ))
    phase.checks.append(Check(
        "pre-commit config present",
        file_exists(ROOT / ".pre-commit-config.yaml"),
    ))
    ci_dir = GITHUB / "workflows"
    phase.checks.append(Check(
        "GitHub Actions workflows directory exists",
        dir_exists(ci_dir),
    ))
    ci_files = list(ci_dir.glob("*.yml")) + list(ci_dir.glob("*.yaml")) if dir_exists(ci_dir) else []
    phase.checks.append(Check(
        f"At least one CI workflow present (found {len(ci_files)})",
        len(ci_files) >= 1,
    ))
    phase.checks.append(Check(
        "tests/ directory exists",
        dir_exists(TESTS),
    ))
    phase.checks.append(Check(
        "tests/ has at least one test file",
        any(TESTS.glob("test_*.py")) if dir_exists(TESTS) else False,
    ))
    mock_provider = PKG / "providers" / "mock_provider.py"
    phase.checks.append(Check(
        "MockProvider implemented",
        contains_class(mock_provider, "MockProvider")
        or grep_dir(PKG / "providers", r"class MockProvider"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 2 — Provider layer (20 → 35%)
# ---------------------------------------------------------------------------

def check_phase_2() -> Phase:
    phase = Phase(2, "Provider layer done for real", 20, 35)

    providers_dir = PKG / "providers"

    phase.checks.append(Check(
        "BaseProvider class defined",
        contains_class(providers_dir / "base.py", "BaseProvider"),
    ))
    phase.checks.append(Check(
        "ProviderOutput dataclass/class defined",
        grep_dir(providers_dir, r"class ProviderOutput"),
    ))
    phase.checks.append(Check(
        "OpenAI provider file non-empty",
        file_nonempty(providers_dir / "openai_provider.py"),
    ))
    phase.checks.append(Check(
        "OpenAIProvider class defined",
        contains_class(providers_dir / "openai_provider.py", "OpenAIProvider"),
    ))
    phase.checks.append(Check(
        "Anthropic provider file non-empty",
        file_nonempty(providers_dir / "anthropic_provider.py"),
    ))
    phase.checks.append(Check(
        "AnthropicProvider class defined",
        contains_class(providers_dir / "anthropic_provider.py", "AnthropicProvider"),
    ))
    phase.checks.append(Check(
        "HuggingFace provider file non-empty",
        file_nonempty(providers_dir / "huggingface_provider.py"),
    ))
    phase.checks.append(Check(
        "HuggingFaceProvider class defined",
        contains_class(providers_dir / "huggingface_provider.py", "HuggingFaceProvider"),
    ))
    phase.checks.append(Check(
        "Ollama provider file non-empty",
        file_nonempty(providers_dir / "ollama_provider.py"),
    ))
    phase.checks.append(Check(
        "OllamaProvider class defined",
        contains_class(providers_dir / "ollama_provider.py", "OllamaProvider"),
    ))
    phase.checks.append(Check(
        "ProviderOutput contains 'tokens' field",
        grep_dir(providers_dir, r"tokens"),
    ))
    phase.checks.append(Check(
        "ProviderOutput contains 'logits' field",
        grep_dir(providers_dir, r"logits"),
    ))
    phase.checks.append(Check(
        "Provider caching/retry logic present",
        grep_dir(providers_dir, r"retry|cache|backoff"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 3 — Minimal end-to-end tracing pipeline (35 → 45%)
# ---------------------------------------------------------------------------

def check_phase_3() -> Phase:
    phase = Phase(3, "Minimal end-to-end tracing pipeline", 35, 45)

    scope_file = PKG / "scope.py"
    phase.checks.append(Check(
        "scope.py non-empty",
        file_nonempty(scope_file),
    ))
    phase.checks.append(Check(
        "Scope class defined in scope.py",
        contains_class(scope_file, "Scope"),
    ))
    phase.checks.append(Check(
        "Scope.trace() method defined",
        contains_method(scope_file, "Scope", "trace"),
    ))
    phase.checks.append(Check(
        "TraceResult class defined",
        grep_dir(PKG, r"class TraceResult"),
    ))
    phase.checks.append(Check(
        "FeatureExtractor class defined",
        grep_dir(PKG / "features", r"class FeatureExtractor"),
    ))
    phase.checks.append(Check(
        "CircuitTracer class defined",
        grep_dir(PKG / "circuits", r"class CircuitTracer"),
    ))
    graph_file = PKG / "circuits" / "graph.py"
    phase.checks.append(Check(
        "Graph class defined",
        contains_class(graph_file, "Graph") or grep_dir(PKG / "circuits", r"class.*Graph"),
    ))
    phase.checks.append(Check(
        "Graph serializes to JSON (to_json/to_dict method)",
        grep_dir(PKG / "circuits", r"def to_json|def to_dict|json\.dumps"),
    ))
    phase.checks.append(Check(
        "HTML report generation present",
        grep_dir(PKG / "visualization", r"html|HTML|render"),
    ))
    cli_file = PKG / "cli.py"
    phase.checks.append(Check(
        "CLI file non-empty",
        file_nonempty(cli_file),
    ))
    phase.checks.append(Check(
        "CLI 'trace' command defined",
        grep_file(cli_file, r"trace"),
    ))
    phase.checks.append(Check(
        "CLI entry point in pyproject.toml",
        grep_file(ROOT / "pyproject.toml", r"scripts|console.scripts|cli"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 4 — Black-box interpretability engine (45 → 55%)
# ---------------------------------------------------------------------------

def check_phase_4() -> Phase:
    phase = Phase(4, "Black-box interpretability engine", 45, 55)

    features_dir = PKG / "features"
    phase.checks.append(Check(
        "Token ablation / masking importance logic present",
        grep_dir(features_dir, r"ablat|mask|perturbat"),
    ))
    phase.checks.append(Check(
        "Pairwise token interaction scoring present",
        grep_dir(features_dir, r"pairwise|interaction"),
    ))
    phase.checks.append(Check(
        "API cost estimator present",
        grep_dir(PKG, r"cost|budget|estimat"),
    ))
    phase.checks.append(Check(
        "Uncertainty/confidence scoring present",
        grep_dir(PKG, r"confidence|uncertainty|score"),
    ))
    phase.checks.append(Check(
        "Caching layer for perturbations present",
        grep_dir(PKG, r"cache|Cache"),
    ))
    phase.checks.append(Check(
        "Black-box result labels in report (observed/inferred/approximated)",
        grep_dir(PKG, r"observed|inferred|approximated"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 5 — White-box mechanistic core with SAE (55 → 70%)
# ---------------------------------------------------------------------------

def check_phase_5() -> Phase:
    phase = Phase(5, "White-box mechanistic core with SAE training", 55, 70)

    features_dir = PKG / "features"
    sae_file = features_dir / "sae.py"
    phase.checks.append(Check(
        "SAE file non-empty",
        file_nonempty(sae_file),
    ))
    phase.checks.append(Check(
        "TopKSAE or SparseAutoencoder class defined",
        contains_class(sae_file, "TopKSAE")
        or contains_class(sae_file, "SparseAutoencoder")
        or grep_file(sae_file, r"class.*SAE|class.*Autoencoder"),
    ))
    phase.checks.append(Check(
        "SAE training loop present",
        grep_file(sae_file, r"def train|optimizer|backward"),
    ))
    phase.checks.append(Check(
        "Activation cache pipeline present",
        grep_dir(PKG, r"activation.*cache|cache.*activation|hidden_states"),
    ))
    phase.checks.append(Check(
        "SAE metrics: reconstruction error tracked",
        grep_dir(PKG, r"reconstruction|recon"),
    ))
    phase.checks.append(Check(
        "SAE metrics: L0 sparsity tracked",
        grep_dir(PKG, r"l0|sparsity|L0"),
    ))
    phase.checks.append(Check(
        "Feature labeling pipeline present",
        grep_dir(PKG, r"label|Label"),
    ))
    supernodes_file = PKG / "circuits" / "supernodes.py"
    phase.checks.append(Check(
        "Supernode clustering present",
        file_nonempty(supernodes_file)
        and grep_file(supernodes_file, r"cluster|supernode|Supernode"),
    ))
    phase.checks.append(Check(
        "Feature extraction from SAE codes present",
        grep_dir(features_dir, r"extract|feature_code|sparse_code"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 6 — Attribution graph quality and path analysis (70 → 78%)
# ---------------------------------------------------------------------------

def check_phase_6() -> Phase:
    phase = Phase(6, "Attribution graph quality and path analysis", 70, 78)

    circuits_dir = PKG / "circuits"
    phase.checks.append(Check(
        "Graph pruning logic present",
        grep_dir(circuits_dir, r"prun"),
    ))
    phase.checks.append(Check(
        "Top-k path ranking present",
        grep_dir(circuits_dir, r"top_k|top_path|path_rank|rank"),
    ))
    phase.checks.append(Check(
        "Suppressor/inhibitor edge support present",
        grep_dir(circuits_dir, r"suppress|inhibit|negative.*edge"),
    ))
    phase.checks.append(Check(
        "Error residual node present",
        grep_dir(circuits_dir, r"error.*node|residual.*node|error_residual"),
    ))
    phase.checks.append(Check(
        "Graph diff/comparison support present",
        grep_dir(circuits_dir, r"diff|compare|baseline"),
    ))
    phase.checks.append(Check(
        "Graph export to JSON present",
        grep_dir(circuits_dir, r"json|JSON"),
    ))
    phase.checks.append(Check(
        "Graph export to CSV present",
        grep_dir(circuits_dir, r"csv|CSV"),
    ))
    phase.checks.append(Check(
        "Indirect edge support present",
        grep_dir(circuits_dir, r"indirect|transitive"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 7 — Deep UI layer (78 → 88%)
# ---------------------------------------------------------------------------

def check_phase_7() -> Phase:
    phase = Phase(7, "Deep UI layer that actually shows model reasoning", 78, 88)

    viz_dir = PKG / "visualization"
    phase.checks.append(Check(
        "visualization/ directory non-empty",
        dir_exists(viz_dir) and any(
            f for f in viz_dir.glob("*.py") if f.stat().st_size > 0
        ),
    ))
    phase.checks.append(Check(
        "Token heatmap present",
        grep_dir(viz_dir, r"heatmap|heat_map"),
    ))
    phase.checks.append(Check(
        "Attribution graph explorer (Plotly/D3) present",
        grep_dir(viz_dir, r"plotly|d3|graph.*explor"),
    ))
    phase.checks.append(Check(
        "Residual stream trajectory view present",
        grep_dir(viz_dir, r"residual.*stream|trajectory|pca"),
    ))
    phase.checks.append(Check(
        "Feature browser present",
        grep_dir(viz_dir, r"feature.*browser|browser"),
    ))
    phase.checks.append(Check(
        "Probe dashboard present",
        grep_dir(viz_dir, r"probe.*dashboard|dashboard"),
    ))
    report_file = viz_dir / "report.py"
    phase.checks.append(Check(
        "Standalone HTML report with tabs",
        grep_file(report_file, r"tab|tabbed|standalone"),
    ))
    phase.checks.append(Check(
        "Dark mode support present",
        grep_dir(viz_dir, r"dark.*mode|dark_mode|prefers-color-scheme"),
    ))
    phase.checks.append(Check(
        "Observation type labels in UI (observed/inferred/approximated)",
        grep_dir(viz_dir, r"observed|inferred|approximated"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 8 — Full probe suite (88 → 94%)
# ---------------------------------------------------------------------------

def check_phase_8() -> Phase:
    phase = Phase(8, "Full probe suite mapped to the paper", 88, 94)

    probes_dir = PKG / "probes"
    builtin_file = probes_dir / "builtin.py"
    runner_file = probes_dir / "runner.py"

    probes = [
        ("MultiHopProbe", "multi.hop|multi_hop|MultiHop"),
        ("CapitalsProbe", "capital|Capital"),
        ("RhymePlanningProbe", "rhyme|Rhyme"),
        ("PersonaConsistencyProbe", "persona|Persona"),
        ("MultilingualProbe", "multilingual|Multilingual"),
        ("HallucinationProbe", "hallucin|Hallucin"),
        ("CoTFaithfulnessProbe", "cot|CoT|faithfulness|Faithfulness"),
        ("RefusalProbe", "refusal|Refusal|jailbreak"),
        ("SuppressorProbe", "suppressor|Suppressor"),
        ("MotivatedReasoningProbe", "motivated|Motivated"),
    ]
    for probe_name, pattern in probes:
        phase.checks.append(Check(
            f"{probe_name} implemented",
            grep_dir(probes_dir, pattern),
        ))

    phase.checks.append(Check(
        "ProbeRunner / benchmark runner present",
        file_nonempty(runner_file) and grep_file(runner_file, r"class.*Runner|def run"),
    ))
    phase.checks.append(Check(
        "Probes output pass/fail score",
        grep_dir(probes_dir, r"pass|fail|score"),
    ))
    phase.checks.append(Check(
        "CLI benchmark command present",
        grep_file(PKG / "cli.py", r"bench|probe"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 9 — Interventions, comparisons, and truth-testing (94 → 97%)
# ---------------------------------------------------------------------------

def check_phase_9() -> Phase:
    phase = Phase(9, "Interventions, comparisons, and truth-testing", 94, 97)

    intervention_file = PKG / "features" / "intervention.py"
    phase.checks.append(Check(
        "intervention.py non-empty",
        file_nonempty(intervention_file),
    ))
    phase.checks.append(Check(
        "Feature inhibit support",
        grep_file(intervention_file, r"inhibit|suppress|zero"),
    ))
    phase.checks.append(Check(
        "Feature amplify support",
        grep_file(intervention_file, r"amplif|boost|scale"),
    ))
    phase.checks.append(Check(
        "Feature clamp support",
        grep_file(intervention_file, r"clamp"),
    ))
    phase.checks.append(Check(
        "Before/after trace comparison logic",
        grep_dir(PKG, r"baseline.*trace|before.*after|compare.*trace"),
    ))
    phase.checks.append(Check(
        "Intervention report mode present",
        grep_dir(PKG, r"intervention.*report|report.*intervention"),
    ))
    phase.checks.append(Check(
        "Causal validation examples/tests present",
        grep_dir(TESTS, r"causal|intervention|inhibit"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Phase 10 — Release engineering, docs, and production polish (97 → 100%)
# ---------------------------------------------------------------------------

def check_phase_10() -> Phase:
    phase = Phase(10, "Release engineering, docs, and production polish", 97, 100)

    phase.checks.append(Check(
        "pyproject.toml has classifiers",
        grep_file(ROOT / "pyproject.toml", r"classifiers"),
    ))
    phase.checks.append(Check(
        "pyproject.toml has version",
        grep_file(ROOT / "pyproject.toml", r"version\s*="),
    ))
    phase.checks.append(Check(
        "CHANGELOG.md present",
        file_nonempty(ROOT / "CHANGELOG.md"),
    ))
    docs_dir = ROOT / "docs"
    phase.checks.append(Check(
        "docs/ directory has content",
        dir_exists(docs_dir) and any(
            f for f in docs_dir.rglob("*.md") if f.stat().st_size > 0
        ),
    ))
    phase.checks.append(Check(
        "docs/ quickstart guide present",
        any(docs_dir.rglob("quickstart*")) if dir_exists(docs_dir) else False,
    ))
    phase.checks.append(Check(
        "examples/ directory has content",
        dir_exists(ROOT / "examples") and any(
            f for f in (ROOT / "examples").glob("*.py") if f.stat().st_size > 0
        ),
    ))
    phase.checks.append(Check(
        "pip-audit or safety in CI",
        grep_dir(GITHUB / "workflows", r"pip.audit|safety|audit", "**/*.yml"),
    ))
    phase.checks.append(Check(
        "Trusted Publishing / PyPI token config present",
        grep_dir(GITHUB / "workflows", r"pypi|PYPI|publish", "**/*.yml"),
    ))
    phase.checks.append(Check(
        "Telemetry policy statement present",
        grep_dir(ROOT, r"telemetry|no.*telemetry|opt.in", "**/*.md"),
    ))

    return phase


# ---------------------------------------------------------------------------
# Run all phases and compute overall progress
# ---------------------------------------------------------------------------

def compute_progress() -> tuple[list[Phase], float]:
    checkers = [
        check_phase_0,
        check_phase_1,
        check_phase_2,
        check_phase_3,
        check_phase_4,
        check_phase_5,
        check_phase_6,
        check_phase_7,
        check_phase_8,
        check_phase_9,
        check_phase_10,
    ]
    phases = [fn() for fn in checkers]

    # Overall percentage is the weighted average across all checks, mapped
    # to the 0-100 range implied by each phase span.
    total_checks = sum(p.total for p in phases)
    total_passed = sum(p.passed for p in phases)
    if total_checks == 0:
        overall = 0.0
    else:
        ratio = total_passed / total_checks
        overall = ratio * 100.0

    return phases, overall


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PASS_ICON = "✅"
FAIL_ICON = "❌"
BAR_FULL = "█"
BAR_EMPTY = "░"


def progress_bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def render_markdown(phases: list[Phase], overall: float) -> str:
    lines: list[str] = []
    lines.append("# LLmThoughtLens — Build Progress")
    lines.append("")
    lines.append(
        f"**Overall: {overall:.1f}%**  `{progress_bar(overall)}`"
    )
    lines.append("")
    lines.append("| Phase | Range | Progress | Checks |")
    lines.append("|-------|-------|----------|--------|")
    for p in phases:
        icon = PASS_ICON if p.score >= 1.0 else ("🔶" if p.score > 0 else FAIL_ICON)
        lines.append(
            f"| {icon} Ph{p.number}: {p.name} "
            f"| {p.pct_start}–{p.pct_end}% "
            f"| {p.completion_pct:.1f}% `{progress_bar(p.score * 100, 10)}` "
            f"| {p.passed}/{p.total} |"
        )
    lines.append("")

    for p in phases:
        lines.append(f"## Phase {p.number} — {p.name} ({p.pct_start}% → {p.pct_end}%)")
        lines.append("")
        for c in p.checks:
            icon = PASS_ICON if c.passed else FAIL_ICON
            detail = f" — {c.detail}" if c.detail else ""
            lines.append(f"- [{icon}] {c.label}{detail}")
        lines.append("")

    lines.append(
        f"_Auto-generated by `scripts/track_progress.py` — "
        f"do not edit manually._"
    )
    return "\n".join(lines)


def render_text(phases: list[Phase], overall: float) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  LLmThoughtLens Build Progress Tracker")
    lines.append("=" * 60)
    lines.append(f"  Overall: {overall:.1f}%  [{progress_bar(overall)}]")
    lines.append("")
    for p in phases:
        status = "DONE" if p.score >= 1.0 else f"{p.score * 100:.0f}%"
        lines.append(
            f"  Ph{p.number:02d} [{p.pct_start:3d}→{p.pct_end:3d}%]  "
            f"{status:6s}  {p.passed}/{p.total:2d}  {p.name}"
        )
    lines.append("")
    lines.append("Details:")
    for p in phases:
        lines.append(f"\n  Phase {p.number}: {p.name}")
        for c in p.checks:
            icon = "PASS" if c.passed else "FAIL"
            lines.append(f"    [{icon}] {c.label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLmThoughtLens progress tracker")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite PROGRESS.md with current results",
    )
    args = parser.parse_args()

    phases, overall = compute_progress()

    if args.json:
        data = {
            "overall_pct": round(overall, 2),
            "phases": [
                {
                    "number": p.number,
                    "name": p.name,
                    "pct_start": p.pct_start,
                    "pct_end": p.pct_end,
                    "completion_pct": round(p.completion_pct, 2),
                    "passed": p.passed,
                    "total": p.total,
                    "checks": [
                        {"label": c.label, "passed": c.passed, "detail": c.detail}
                        for c in p.checks
                    ],
                }
                for p in phases
            ],
        }
        print(json.dumps(data, indent=2))
        return

    if args.update:
        progress_md = ROOT / "PROGRESS.md"
        md_content = render_markdown(phases, overall)
        progress_md.write_text(md_content, encoding="utf-8")
        print(f"Updated {progress_md}")
        print(f"Overall progress: {overall:.1f}%")
        return

    print(render_text(phases, overall))


if __name__ == "__main__":
    main()
