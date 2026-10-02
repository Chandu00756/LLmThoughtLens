"""Tests for LLmThoughtLens.bench (runner, schema, metrics, scorecards) and the
Ollama provider's robustness features (thinking models, logprobs gating,
retries) with mocked HTTP.

Everything is offline: models are the mock provider, scripted black-box
providers, the tiny random GPT-2 (skipped without torch) or an
:class:`OllamaProvider` wired to an ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest
from LLmThoughtLens.bench import (
    DEFAULT_METRICS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    AttributionFaithfulness,
    BenchConfig,
    BenchResult,
    FirstTokenConfidence,
    Metric,
    MetricContext,
    MetricSkipped,
    ModelSpec,
    aggregate,
    capture_environment,
    evaluate_metric,
    get_metric,
    git_info,
    list_metrics,
    parse_spec,
    parse_specs,
    register_metric,
    render_html,
    render_markdown,
    run_and_write,
    run_benchmark,
    unregister_metric,
    validate_result,
    write_reports,
)
from LLmThoughtLens.probes.base import BaseProbe, ProbeResult, complete
from LLmThoughtLens.probes.builtin import MultiHopProbe
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider
from LLmThoughtLens.providers.ollama_provider import (
    LOGPROBS_MIN_VERSION,
    OllamaProvider,
    looks_like_thinking_model,
    parse_version,
    strip_think_blocks,
)

FAST = {"capture_env": False}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Scripted(BaseProvider):
    """Black-box provider with a fixed answer, real-or-absent logprobs and usage."""

    evidence_kind = "black_box"

    def __init__(
        self, answer: str = "Austin", *, logprob: float | None = None, cost: float | None = None
    ) -> None:
        self.answer = answer
        self.logprob = logprob
        self.cost = cost
        self.n_calls = 0

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model_id(self) -> str:
        return "scripted/v1"

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        self.n_calls += 1
        meta: dict[str, Any] = {
            "completion": self.answer,
            "has_logprobs": self.logprob is not None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
        if self.cost is not None:
            meta["api_cost_usd"] = self.cost
        prob = math.exp(self.logprob) if self.logprob is not None else 1.0
        return ProviderOutput(
            prompt=prompt,
            tokens=self.answer.split(),
            top_tokens=[(self.answer.split()[0], prob)],
            evidence_kind="black_box",
            meta=meta,
        )


class BoomProbe(BaseProbe):
    name = "boom"
    description = "always raises"
    threshold = "never passes"

    def run(self, provider: BaseProvider, prompt: str | None = None) -> ProbeResult:
        raise RuntimeError("probe exploded")


class BadScoreProbe(BaseProbe):
    name = "bad_score"

    def run(self, provider: BaseProvider, prompt: str | None = None) -> ProbeResult:
        return ProbeResult(self.name, score=1.5, passed=True)


class HtmlProbe(BaseProbe):
    """Puts markup in its summary to check escaping."""

    name = "html_probe"
    description = "<i>desc</i>"

    def run(self, provider: BaseProvider, prompt: str | None = None) -> ProbeResult:
        c = complete(provider, "x")
        return ProbeResult(self.name, score=0.5, passed=False, summary=f"<script>{c.text}</script>")


class PlainProbe(BaseProbe):
    """A custom probe that does not call annotate()."""

    name = "plain"

    def run(self, provider: BaseProvider, prompt: str | None = None) -> ProbeResult:
        provider.run("x")
        return ProbeResult(self.name, score=1.0, passed=True, summary="fine")


def scripted_spec(label: str = "scripted", **kw: Any) -> ModelSpec:
    return ModelSpec("custom", label=label, factory=lambda: Scripted(**kw))


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


class TestSpecs:
    def test_parse_aliases_and_labels(self):
        s = parse_spec("hf:gpt2")
        assert (s.provider, s.model, s.label) == ("huggingface", "gpt2", "hf/gpt2")
        o = ModelSpec.parse("ollama:qwen3:1.7b")
        assert (o.provider, o.model, o.label) == ("ollama", "qwen3:1.7b", "ollama/qwen3:1.7b")
        assert ModelSpec.parse("claude:claude-haiku-4-5").provider == "anthropic"
        m = ModelSpec.parse(" mock ")
        assert (m.provider, m.model, m.label) == ("mock", "", "mock")
        assert m.is_local and not ModelSpec.parse("openai:gpt-4o-mini").is_local

    def test_parse_specs_string_and_duplicates(self):
        specs = parse_specs("mock, hf:gpt2")
        assert [s.label for s in specs] == ["mock", "hf/gpt2"]
        with pytest.raises(ValueError, match="duplicate"):
            parse_specs(["mock", "mock"])
        with pytest.raises(ValueError):
            parse_specs([])
        with pytest.raises(ValueError):
            ModelSpec.parse("   ")

    def test_secrets_redacted(self):
        s = ModelSpec.parse("openai:gpt-4o-mini", api_key="sk-secret", timeout=3, obj=object())
        d = s.as_dict()
        assert d["kwargs"]["api_key"] == "***"
        assert d["kwargs"]["timeout"] == 3
        assert isinstance(d["kwargs"]["obj"], str)
        assert "sk-secret" not in json.dumps(d)

    def test_build_mock_and_factory(self):
        assert isinstance(ModelSpec.parse("mock").build(), MockProvider)
        spec = scripted_spec()
        assert isinstance(spec.build(), Scripted)
        assert spec.as_dict()["custom_factory"] is True

    def test_build_ollama_defaults_to_version_gated_logprobs(self):
        p = ModelSpec.parse("ollama:llama3.1:8b").build()
        assert isinstance(p, OllamaProvider)
        assert p.request_logprobs == "auto" and p.model == "llama3.1:8b"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunnerOnMock:
    def test_schema_and_synthetic_labelling(self):
        res = run_benchmark(["mock"], ["multi_hop", "capitals"], repeats=2, **FAST)
        assert res.validate() == []
        d = res.data
        assert d["schema"] == SCHEMA_NAME and d["schema_version"] == SCHEMA_VERSION
        assert len(res.cells) == 4
        assert res.models[0]["synthetic"] is True
        assert all(c["synthetic"] for c in res.cells)
        assert all(c["summary"].startswith("[synthetic") for c in res.cells)
        assert any("Synthetic providers" in n for n in d["notes"])
        assert any("temperature=0" in n for n in d["notes"])
        assert [c["seed"] for c in res.cells] == [0, 0, 1, 1]
        assert res.model_labels() == ["mock"] and res.probe_names() == ["multi_hop", "capitals"]
        agg = res.aggregate("mock", "capitals")
        assert agg is not None and agg["synthetic"] is True and agg["n"] == 2
        assert res.aggregate("mock", "nope") is None
        assert res.environment == {}
        # Synthetic providers expose no real probabilities.
        ftc = res.cells[0]["metrics"]["first_token_confidence"]
        assert ftc["status"] == "skipped"

    def test_deterministic_across_runs(self):
        a = run_benchmark(["mock"], None, **FAST)
        b = run_benchmark(["mock"], None, **FAST)
        key = [(c["probe"], c["score"], c["passed"], c["summary"]) for c in a.cells]
        assert key == [(c["probe"], c["score"], c["passed"], c["summary"]) for c in b.cells]
        assert len(a.cells) == 10
        assert all(r["consistent"] for r in a.aggregates["by_model_probe"])

    def test_explicit_seeds_and_config_object(self):
        cfg = BenchConfig(seeds=[5, 9], temperature=0.7, capture_env=False, include_evidence=False)
        res = run_benchmark(["mock"], ["multi_hop"], cfg)
        assert [c["seed"] for c in res.cells] == [5, 9]
        assert res.data["config"]["deterministic_decoding"] is False
        assert all(c["evidence"] is None for c in res.cells)
        assert not any("temperature=0" in n for n in res.data["notes"])

    def test_invalid_arguments_raise(self):
        with pytest.raises(KeyError):
            run_benchmark(["mock"], ["no_such_probe"], **FAST)
        with pytest.raises(ValueError):
            run_benchmark(["mock"], [MultiHopProbe(), MultiHopProbe()], **FAST)
        with pytest.raises(ValueError):
            run_benchmark(["mock"], [], **FAST)
        with pytest.raises(ValueError):
            run_benchmark(["mock"], repeats=0, **FAST)
        with pytest.raises(ValueError):
            run_benchmark(["mock"], seeds=[], **FAST)
        with pytest.raises(KeyError):
            run_benchmark(["mock"], metrics=["nope"], **FAST)

    def test_progress_events_and_broken_callback(self):
        events: list[dict[str, Any]] = []
        run_benchmark(["mock"], ["multi_hop"], progress=events.append, **FAST)
        kinds = [e["event"] for e in events]
        assert kinds == ["model_start", "model_ready", "cell_start", "cell_done", "model_done"]
        done = events[3]
        assert done["index"] == 1 and done["total"] == 1 and done["status"] == "ok"

        def broken(_e: dict[str, Any]) -> None:
            raise RuntimeError("ui crashed")

        assert run_benchmark(["mock"], ["multi_hop"], progress=broken, **FAST).validate() == []

    def test_environment_captured_by_default(self):
        res = run_benchmark(["mock"], ["multi_hop"], include_devices=False)
        env = res.environment
        assert env["python"] and "numpy" in env["packages"]
        assert "devices" not in env
        assert "commit" in env["git"]


class TestFailureIsolation:
    def test_failing_model_does_not_abort_run(self):
        def explode() -> BaseProvider:
            raise ImportError("missing extra")

        bad = ModelSpec("custom", label="broken", factory=explode)
        res = run_benchmark([bad, "mock"], ["multi_hop", "capitals"], repeats=2, **FAST)
        assert res.validate() == []
        broken = res.models[0]
        assert broken["status"] == "error"
        assert (
            broken["error"]["type"] == "ImportError" and broken["error"]["stage"] == "provider_init"
        )
        bad_cells = [c for c in res.cells if c["model"] == "broken"]
        assert len(bad_cells) == 4 and all(c["status"] == "error" for c in bad_cells)
        assert res.models[1]["status"] == "ok"
        assert all(c["status"] == "ok" for c in res.cells if c["model"] == "mock")
        by_model = {r["model"]: r for r in res.aggregates["by_model"]}
        assert by_model["broken"]["n_error_cells"] == 4 and by_model["broken"]["mean_score"] is None
        assert any("failed to load" in n for n in res.data["notes"])

    def test_failing_probe_fails_only_its_cell(self):
        res = run_benchmark(["mock"], [BoomProbe(), MultiHopProbe(), BadScoreProbe()], **FAST)
        assert res.validate() == []
        status = {c["probe"]: c for c in res.cells}
        assert status["boom"]["status"] == "error"
        assert status["boom"]["error"]["message"] == "probe exploded"
        assert "RuntimeError" in status["boom"]["summary"]
        assert status["bad_score"]["status"] == "error"
        assert status["multi_hop"]["status"] == "ok"
        agg = res.aggregate("mock", "boom")
        assert agg is not None and agg["passed"] is None and agg["n_error"] == 1

    def test_warmup_failure_is_a_model_error(self):
        class DeadServer(Scripted):
            def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
                raise httpx.ConnectError("refused")

        spec = ModelSpec("ollama", "x", label="dead", factory=DeadServer)
        res = run_benchmark([spec], ["multi_hop"], **FAST)
        assert res.models[0]["status"] == "error"
        assert res.models[0]["error"]["type"] == "ConnectError"

    def test_custom_probe_on_synthetic_provider_is_labelled(self):
        res = run_benchmark(["mock"], [PlainProbe()], **FAST)
        cell = res.cells[0]
        assert cell["synthetic"] is True
        assert cell["summary"].startswith("[synthetic provider")


class TestUsageCostAndMetrics:
    def test_usage_and_price_table(self):
        spec = scripted_spec(logprob=math.log(0.5))
        res = run_benchmark(
            [spec], ["multi_hop", "capitals"], prices={"scripted": (1.0, 2.0)}, **FAST
        )
        mh = next(c for c in res.cells if c["probe"] == "multi_hop")
        assert mh["usage"]["calls"] == 1
        assert mh["usage"]["prompt_tokens"] == 10 and mh["usage"]["completion_tokens"] == 4
        assert mh["cost_usd"] == pytest.approx((10 * 1.0 + 4 * 2.0) / 1e6)
        model = res.models[0]
        assert model["usage"]["calls"] == 11
        assert model["cost_usd"] == pytest.approx(11 * 18 / 1e6)
        ftc = mh["metrics"]["first_token_confidence"]
        assert ftc["status"] == "ok"
        assert ftc["value"] == {"mean": pytest.approx(0.5), "n": 1, "of": 1}
        assert res.models[0]["evidence_kind"] == "black_box"
        assert res.models[0]["framing"] == "chat"

    def test_provider_reported_cost_wins_and_no_price_means_none(self):
        res = run_benchmark([scripted_spec(cost=0.01)], ["multi_hop"], **FAST)
        assert res.cells[0]["cost_usd"] == pytest.approx(0.01)
        res2 = run_benchmark([scripted_spec()], ["multi_hop"], **FAST)
        assert res2.cells[0]["cost_usd"] is None and res2.models[0]["cost_usd"] is None
        # No logprobs: the placeholder is never reported as confidence.
        assert res2.cells[0]["metrics"]["first_token_confidence"]["status"] == "skipped"

    def test_attribution_faithfulness_skipped_without_white_box(self):
        res = run_benchmark([scripted_spec()], ["multi_hop"], **FAST)
        rec = res.models[0]["metrics"]["attribution_faithfulness"]
        assert rec["status"] == "skipped" and "white-box" in rec["reason"]

    def test_attribution_faithfulness_adapter(self):
        class WhiteBox(Scripted):
            supports_gradients = True

        calls: list[str] = []

        def fn(provider: Any, prompt: str) -> float:
            calls.append(prompt)
            return 0.25 if "Dallas" in prompt else 0.75

        register_metric(AttributionFaithfulness(fn=fn), replace=True)
        try:
            spec = ModelSpec("custom", label="wb", factory=WhiteBox)
            res = run_benchmark([spec], ["multi_hop"], **FAST)
        finally:
            register_metric(AttributionFaithfulness(), replace=True)
        rec = res.models[0]["metrics"]["attribution_faithfulness"]
        assert rec["status"] == "ok"
        assert rec["value"]["mean"] == pytest.approx(0.5)
        assert len(calls) == 2

    def test_attribution_faithfulness_mapping_and_bad_values(self):
        class WB(Scripted):
            supports_gradients = True

        ctx = MetricContext(level="model", spec=scripted_spec(), provider=WB())
        mapping = AttributionFaithfulness(fn=lambda p, q: {"spearman": 0.5, "pearson": None})
        rec = evaluate_metric(mapping, ctx)
        assert rec["status"] == "ok"
        assert rec["value"]["mean"] == {"pearson": None, "spearman": 0.5}
        bad = AttributionFaithfulness(fn=lambda p, q: "nope")
        assert evaluate_metric(bad, ctx)["status"] == "error"

    def test_default_faithfulness_resolution_is_graceful(self, monkeypatch):
        class WB(Scripted):
            supports_gradients = True

        ctx = MetricContext(level="model", spec=scripted_spec(), provider=WB())
        metric = AttributionFaithfulness()
        monkeypatch.setattr(AttributionFaithfulness, "MODULE", "LLmThoughtLens._no_such_module")
        rec = evaluate_metric(metric, ctx)
        assert rec["status"] == "skipped" and "not importable" in rec["reason"]
        monkeypatch.setattr(AttributionFaithfulness, "MODULE", "LLmThoughtLens.probes.base")
        rec = evaluate_metric(metric, ctx)
        assert rec["status"] == "skipped" and "register an adapter" in rec["reason"]
        assert evaluate_metric(metric, MetricContext("model", scripted_spec(), None))["status"] == (
            "skipped"
        )

    def test_registry_api(self):
        assert set(DEFAULT_METRICS) <= set(list_metrics())
        assert isinstance(get_metric("first_token_confidence"), FirstTokenConfidence)
        with pytest.raises(KeyError):
            get_metric("nope")
        with pytest.raises(ValueError, match="already registered"):
            register_metric(FirstTokenConfidence())

        class Nameless(Metric):
            def compute(self, ctx: MetricContext) -> Any:
                return 1

        with pytest.raises(ValueError):
            register_metric(Nameless())

        class BadLevel(Metric):
            name = "bad_level"
            level = "run"  # type: ignore[assignment]

            def compute(self, ctx: MetricContext) -> Any:
                return 1

        with pytest.raises(ValueError):
            register_metric(BadLevel())
        unregister_metric("not-registered")  # no error

    def test_custom_metrics_never_change_the_schema(self):
        class CellLen(Metric):
            name = "summary_len"
            level = "cell"
            description = "len(summary)"

            def compute(self, ctx: MetricContext) -> Any:
                assert ctx.result is not None
                return {"len": len(ctx.result.summary), "nan": float("nan")}

        class Skipper(Metric):
            name = "skipper"
            level = "model"

            def compute(self, ctx: MetricContext) -> Any:
                raise MetricSkipped("not today")

        class NoneMetric(Metric):
            name = "none_metric"
            level = "model"

            def compute(self, ctx: MetricContext) -> Any:
                return None

        for m in (CellLen(), Skipper(), NoneMetric()):
            register_metric(m, replace=True)
        try:
            res = run_benchmark(
                ["mock"], ["multi_hop"], metrics=["summary_len", "skipper", "none_metric"], **FAST
            )
        finally:
            for n in ("summary_len", "skipper", "none_metric"):
                unregister_metric(n)
        assert res.validate() == []
        rec = res.cells[0]["metrics"]["summary_len"]
        assert rec["status"] == "ok" and rec["value"]["nan"] is None
        assert res.models[0]["metrics"]["skipper"] == {
            "status": "skipped",
            "value": None,
            "reason": "not today",
        }
        assert res.models[0]["metrics"]["none_metric"]["status"] == "skipped"
        assert res.data["metrics"]["levels"] == {
            "summary_len": "cell",
            "skipper": "model",
            "none_metric": "model",
        }


class TestAggregate:
    def test_majority_vote_and_stats(self):
        models = [{"label": "m", "status": "ok", "synthetic": False, "duration_s": 1.0}]
        cells = [
            {"model": "m", "probe": "p", "status": "ok", "score": s, "passed": ok}
            for s, ok in ((1.0, True), (0.0, False), (1.0, True))
        ] + [{"model": "m", "probe": "p", "status": "error", "score": None, "passed": None}]
        agg = aggregate(models, cells, ["p", "q"])
        row = agg["by_model_probe"][0]
        assert row["n"] == 4 and row["n_ok"] == 3 and row["n_error"] == 1
        assert row["mean_score"] == pytest.approx(2 / 3)
        assert row["std_score"] == pytest.approx(math.sqrt(2 / 9))
        assert row["pass_rate"] == pytest.approx(2 / 3) and row["passed"] is True
        assert row["consistent"] is False
        empty = agg["by_model_probe"][1]
        assert empty["mean_score"] is None and empty["passed"] is None
        bm = agg["by_model"][0]
        assert bm["n_scored"] == 1 and bm["n_passed"] == 1 and bm["pass_rate"] == 1.0

    def test_tie_is_not_a_pass(self):
        models = [{"label": "m", "status": "ok"}]
        cells = [
            {"model": "m", "probe": "p", "status": "ok", "score": 1.0, "passed": True},
            {"model": "m", "probe": "p", "status": "ok", "score": 0.0, "passed": False},
        ]
        assert aggregate(models, cells, ["p"])["by_model_probe"][0]["passed"] is False


# ---------------------------------------------------------------------------
# Schema and persistence
# ---------------------------------------------------------------------------


class TestSchema:
    def test_roundtrip(self, tmp_path):
        res = run_benchmark(["mock"], ["multi_hop"], **FAST)
        path = tmp_path / "r.json"
        res.to_json(path)
        again = BenchResult.from_json(path)
        assert again.data == json.loads(res.to_json())
        assert BenchResult.from_json(res.to_json()).cells == res.cells

    def test_problems_are_reported(self):
        good = run_benchmark(["mock"], ["multi_hop"], **FAST).data
        assert validate_result([]) == ["record is not a JSON object"]
        broken = json.loads(json.dumps(good))
        broken["schema_version"] = 99
        del broken["title"]
        broken["cells"][0]["score"] = 7
        broken["cells"][0]["probe"] = "ghost"
        broken["models"][0]["metrics"] = {"x": {"status": "weird"}}
        problems = validate_result(broken)
        text = " | ".join(problems)
        assert "schema_version" in text and "'title'" in text
        assert "not in [0, 1]" in text and "ghost" in text and "valid status" in text
        with pytest.raises(ValueError, match="invalid benchmark record"):
            BenchResult.from_json(json.dumps(broken))

    def test_error_cell_needs_error_record(self):
        data = run_benchmark(["mock"], ["multi_hop"], **FAST).data
        data["cells"][0]["status"] = "error"
        assert any("no error record" in p for p in validate_result(data))
        data["cells"][0]["status"] = "maybe"
        assert any("status" in p for p in validate_result(data))
        del data["cells"][0]["seed"]
        assert any("missing" in p for p in validate_result(data))


class TestEnvironment:
    def test_capture_without_devices(self):
        env = capture_environment(include_devices=False)
        for key in ("python", "platform", "packages", "git", "captured_utc", "cpu_count"):
            assert key in env
        assert env["packages"]["numpy"]
        json.dumps(env, allow_nan=False)

    def test_git_info_outside_a_checkout(self, tmp_path):
        assert git_info(tmp_path) == {"commit": None, "dirty": None}

    def test_devices_probe(self):
        pytest.importorskip("torch")
        env = capture_environment(include_devices=True)
        assert env["devices"]["torch_available"] is True
        assert env["devices"]["default_device"] in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# Scorecards
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_result() -> BenchResult:
    def explode() -> BaseProvider:
        raise RuntimeError("cannot load <weights>")

    bad = ModelSpec("custom", label="broken", factory=explode)
    return run_benchmark(
        ["mock", scripted_spec("scripted", answer="Austin <b>bold</b>"), bad],
        [MultiHopProbe(), BoomProbe(), HtmlProbe()],
        repeats=2,
        capture_env=False,
    )


class TestScorecards:
    def test_markdown(self, mixed_result):
        md = render_markdown(mixed_result)
        assert md.startswith("# LLmThoughtLens probe benchmark")
        assert "| Probe | Style | mock (synthetic) | scripted | broken |" in md
        assert "mock (synthetic)" in md and "broken (failed to load)" in md
        assert "## Errors" in md and "probe exploded" in md and "cannot load" in md
        assert "## Metrics" in md and "first_token_confidence" in md
        assert "1.00 PASS ±0.00 (1.00 pass)" in md
        assert "environment**: not captured" in md

    def test_mapping_metric_is_summarised(self):
        from LLmThoughtLens.bench.scorecard import _metric_text

        rec = {
            "status": "ok",
            "value": {
                "mean": {"pearson": 0.25, "spearman": 0.5, "n": 10.0, "runtime_s": 1.0, "x": None},
                "per_prompt": [{}, {}],
            },
        }
        assert _metric_text(rec) == "spearman 0.500, pearson 0.250, x n/a (mean of 2)"
        assert _metric_text({"status": "ok", "value": {"mean": {"n": 1}}}) == "ok"
        assert _metric_text({"status": "ok", "value": [1, 2]}) == "[1, 2]"
        assert _metric_text({"status": "ok", "value": 0.5}) == "0.500"
        assert _metric_text({"status": "skipped", "reason": "why"}) == "skipped: why"
        assert _metric_text(None) == "n/a"

    def test_markdown_from_plain_dict(self, mixed_result):
        assert render_markdown(mixed_result.data) == render_markdown(mixed_result)

    def test_html_escapes_model_output(self, mixed_result):
        page = render_html(mixed_result, plotlyjs="none")
        assert page.startswith("<!doctype html>")
        assert "<script>Austin" not in page
        assert "&lt;script&gt;Austin &lt;b&gt;bold&lt;/b&gt;&lt;/script&gt;" in page
        assert "&lt;i&gt;desc&lt;/i&gt;" in page
        assert "cannot load &lt;weights&gt;" in page
        assert '<span class="tag">synthetic</span>' in page
        assert "failed to load" in page
        assert "Plotly" not in page
        assert "prefers-color-scheme: dark" in page

    def test_chart_modes(self, mixed_result):
        svg = render_html(mixed_result, plotlyjs="svg")
        assert "<svg" in svg and "Plotly" not in svg and "no successful cells" in svg
        cdn = render_html(mixed_result, plotlyjs="cdn")
        assert "https://cdn.plot.ly/plotly-" in cdn and "Plotly.newPlot" in cdn
        with pytest.raises(ValueError):
            render_html(mixed_result, plotlyjs="png")  # type: ignore[arg-type]

    def test_inline_plotly_is_self_contained(self):
        res = run_benchmark(["mock"], ["multi_hop"], **FAST)
        page = render_html(res)
        assert "Plotly.newPlot" in page and "<script src=" not in page
        assert len(page) > 1_000_000  # plotly.js embedded

    def test_write_reports_and_run_and_write(self, tmp_path, mixed_result):
        paths = write_reports(mixed_result, tmp_path / "a", plotlyjs="none")
        assert sorted(paths) == ["html", "json", "md"]
        assert BenchResult.from_json(paths["json"]).validate() == []
        only_md = write_reports(mixed_result, tmp_path / "b", formats=["md"], stem="x")
        assert list(only_md) == ["md"] and only_md["md"].name == "x.md"
        with pytest.raises(ValueError):
            write_reports(mixed_result, tmp_path / "c", formats=["pdf"])
        res, out = run_and_write(["mock"], tmp_path / "d", ["multi_hop"], plotlyjs="svg", **FAST)
        assert out["html"].read_text().count("<svg") == 1
        assert json.loads(out["json"].read_text())["run_id"] == res.data["run_id"]


# ---------------------------------------------------------------------------
# White-box models (tiny random GPT-2)
# ---------------------------------------------------------------------------


class TestWhiteBoxModel:
    def test_tiny_gpt2_records_model_info(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from tests._tiny_hf import make_provider

        spec = ModelSpec("huggingface", "tiny", label="tiny", factory=make_provider)
        res = run_benchmark([spec], ["multi_hop", "capitals"], **FAST)
        assert res.validate() == []
        info = res.models[0]["model_info"]
        assert info["family"] == "gpt2" and info["n_layers"] == 2 and info["n_params"] > 0
        assert res.models[0]["supports_gradients"] is True
        assert res.models[0]["framing"] == "completion"
        cell = res.cells[0]
        assert cell["synthetic"] is False and cell["completion_source"] == "generate"
        assert cell["usage"]["completion_tokens"] is not None
        assert cell["metrics"]["first_token_confidence"]["status"] == "ok"
        rec = res.models[0]["metrics"]["attribution_faithfulness"]
        assert rec["status"] in ("ok", "skipped", "error")


# ---------------------------------------------------------------------------
# Ollama provider (mocked HTTP)
# ---------------------------------------------------------------------------


class FakeOllama:
    """Routes requests by path; records every request body."""

    def __init__(
        self,
        *,
        version: str | None = "0.34.4",
        caps: list[str] | None = None,
        generate: Any = None,
    ) -> None:
        self.version = version
        self.caps = caps
        self.generate = generate or {"response": "Paris"}
        self.requests: list[tuple[str, dict[str, Any] | None]] = []
        self.fail: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        path = request.url.path
        if path == "/api/version":
            if self.version is None:
                return httpx.Response(404, text="404 page not found")
            return httpx.Response(200, json={"version": self.version})
        if path == "/api/show":
            if self.caps is None:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(
                200,
                json={
                    "capabilities": self.caps,
                    "details": {"family": "qwen3", "parameter_size": "2.0B", "x": 1},
                },
            )
        if path == "/api/generate":
            if self.fail:
                return httpx.Response(self.fail.pop(0), json={"error": "busy"})
            gen = self.generate(body) if callable(self.generate) else self.generate
            return httpx.Response(200, json=gen)
        return httpx.Response(404)

    def generate_bodies(self) -> list[dict[str, Any]]:
        return [b for p, b in self.requests if p == "/api/generate" and b is not None]


def ollama(fake: FakeOllama, model: str = "llama3.1:8b", **kw: Any) -> OllamaProvider:
    kw.setdefault("retry_backoff", 0.0)
    return OllamaProvider(
        model=model, base_url="http://ollama.test", transport=httpx.MockTransport(fake), **kw
    )


def lp(token: str, p: float) -> dict[str, Any]:
    return {
        "token": token,
        "logprob": math.log(p),
        "top_logprobs": [{"token": token, "logprob": math.log(p)}],
    }


class TestOllamaHelpers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0.12.11", (0, 12, 11)),
            ("v0.9.0-rc1", (0, 9, 0)),
            ("1", (1, 0, 0)),
            ("0.0.0", None),
            ("garbage", None),
            (None, None),
        ],
    )
    def test_parse_version(self, text, expected):
        assert parse_version(text) == expected

    def test_thinking_name_heuristic(self):
        assert looks_like_thinking_model("qwen3:1.7b")
        assert looks_like_thinking_model("library/deepseek-r1:8b")
        assert not looks_like_thinking_model("llama3.1:8b")
        assert not looks_like_thinking_model("qwen3-coder:30b")

    def test_strip_think_blocks(self):
        assert strip_think_blocks("plain  ") == ("plain  ", "", False)
        assert strip_think_blocks("<think>a</think>\nB") == ("B", "a", False)
        assert strip_think_blocks("<thinking>x</thinking>Y<think>z</think>W") == (
            "YW",
            "x\nz",
            False,
        )
        assert strip_think_blocks("reasoning</think>Answer") == ("Answer", "reasoning", False)
        assert strip_think_blocks("Ans<think>cut off") == ("Ans", "cut off", True)


class TestOllamaThinking:
    def test_think_false_sent_for_thinking_family(self):
        fake = FakeOllama()
        ollama(fake, "qwen3:1.7b").run("q")
        assert fake.generate_bodies()[0]["think"] is False

    def test_think_omitted_for_non_thinking_model(self):
        fake = FakeOllama()
        ollama(fake, "llama3.1:8b").run("q")
        assert "think" not in fake.generate_bodies()[0]

    def test_capabilities_override_name_heuristic(self):
        fake = FakeOllama(caps=["completion", "thinking"])
        p = ollama(fake, "my-custom-model")
        info = p.server_info()
        assert info["thinking_model"] is True and info["logprobs_supported"] is True
        assert info["model_details"] == {"family": "qwen3", "parameter_size": "2.0B"}
        p.run("q")
        assert fake.generate_bodies()[0]["think"] is False
        fake2 = FakeOllama(caps=["completion"])
        p2 = ollama(fake2, "qwen3:1.7b")
        p2.show()
        p2.run("q")
        assert "think" not in fake2.generate_bodies()[0]

    def test_old_server_gets_no_think_field(self):
        fake = FakeOllama(version="0.8.0")
        p = ollama(fake, "qwen3:1.7b")
        assert p.server_info()["think_supported"] is False
        p.run("q")
        assert "think" not in fake.generate_bodies()[0]

    def test_explicit_think_passthrough_and_caller_override(self):
        fake = FakeOllama()
        ollama(fake, "llama3.1:8b", think="high").run("q")
        ollama(fake, "qwen3:1.7b").run("q", think=True)
        bodies = fake.generate_bodies()
        assert bodies[0]["think"] == "high" and bodies[1]["think"] is True

    def test_inline_think_block_is_stripped(self):
        fake = FakeOllama(
            generate={"response": "<think>let me see</think>\n\nAustin", "eval_count": 9}
        )
        p = ollama(fake, "llama3.1:8b")
        out = p.run("q")
        assert out.meta["completion"] == "Austin"
        assert out.meta["thinking"] == "let me see"
        assert out.meta["thinking_stripped"] is True
        assert out.meta["raw_completion"].startswith("<think>")
        assert out.tokens == ["Austin"]
        # Seen thinking once: later calls disable it.
        p.run("q2")
        assert fake.generate_bodies()[1]["think"] is False

    def test_budget_spent_thinking_is_flagged(self):
        fake = FakeOllama(generate={"response": "", "thinking": "long reasoning"})
        out = ollama(fake, "qwen3:1.7b").run("q")
        assert out.meta["completion"] == ""
        assert out.meta["thinking_truncated"] is True
        assert out.meta["has_logprobs"] is False

    def test_probe_completion_never_contains_thinking(self):
        fake = FakeOllama(generate={"response": "<think>Dallas is in Texas</think>Austin"})
        res = MultiHopProbe().run(ollama(fake))
        assert res.passed and res.evidence["response"] == "Austin"
        assert res.evidence["calls"][0]["thinking_chars"] == len("Dallas is in Texas")


class TestOllamaLogprobs:
    def test_logprobs_present_are_real(self):
        fake = FakeOllama(
            generate={"response": "Paris.", "logprobs": [lp("Paris", 0.6), lp(".", 0.9)]}
        )
        out = ollama(fake).run("q")
        assert out.meta["has_logprobs"] is True
        assert out.top_tokens[0] == ("Paris", pytest.approx(0.6))
        assert out.meta["token_logprobs"] == [
            ["Paris", pytest.approx(math.log(0.6))],
            [".", pytest.approx(math.log(0.9))],
        ]
        assert "logprobs_unavailable_reason" not in out.meta
        c = complete(ollama(fake), "q")
        assert c.first_token_prob == pytest.approx(0.6)
        assert c.token_logprobs is not None and len(c.token_logprobs) == 2

    def test_logprobs_absent_are_never_invented(self):
        fake = FakeOllama(generate={"response": "Paris"})
        out = ollama(fake).run("q")
        assert out.meta["has_logprobs"] is False
        assert out.meta["logprobs_unavailable_reason"] == "not_returned"
        assert "placeholder" in out.meta["evidence_note"]
        assert "token_logprobs" not in out.meta
        assert complete(ollama(fake), "q").first_token_prob is None

    def test_auto_mode_gates_on_server_version(self):
        old = FakeOllama(version="0.12.10")
        out = ollama(old, request_logprobs="auto").run("q")
        assert "logprobs" not in old.generate_bodies()[0]
        assert out.meta["logprobs_unavailable_reason"] == "server_too_old"
        assert out.meta["server_version"] == "0.12.10"
        new = FakeOllama(version=".".join(map(str, LOGPROBS_MIN_VERSION)))
        ollama(new, request_logprobs="auto").run("q")
        assert new.generate_bodies()[0]["logprobs"] is True
        # Unknown version (endpoint missing): ask anyway, never fail.
        unknown = FakeOllama(version=None)
        p = ollama(unknown, request_logprobs="auto")
        p.run("q")
        assert unknown.generate_bodies()[0]["logprobs"] is True
        assert p.server_version() is None and p.server_info()["logprobs_supported"] is None

    def test_disabled_and_invalid_setting(self):
        fake = FakeOllama()
        out = ollama(fake, request_logprobs=False).run("q")
        assert "logprobs" not in fake.generate_bodies()[0]
        assert out.meta["logprobs_unavailable_reason"] == "disabled"
        with pytest.raises(ValueError):
            ollama(fake, request_logprobs="sometimes")  # type: ignore[arg-type]

    def test_logprobs_realigned_past_thinking_tokens(self):
        stream = [
            lp("<think>", 0.99),
            lp("hmm", 0.5),
            lp("</think>", 0.99),
            lp("\n\n", 0.9),
            lp("Austin", 0.7),
        ]
        fake = FakeOllama(generate={"response": "Austin", "thinking": "hmm", "logprobs": stream})
        out = ollama(fake, "qwen3:1.7b", think=True).run("q")
        assert out.meta["has_logprobs"] is True
        assert out.top_tokens[0] == ("Austin", pytest.approx(0.7))
        assert out.meta["token_logprobs"][0][0] == "Austin"
        assert "first answer token" in out.meta["evidence_note"]

    def test_unterminated_thinking_reports_no_probabilities(self):
        stream = [lp("<think>", 0.99), lp("still thinking", 0.5)]
        fake = FakeOllama(
            generate={"response": "", "thinking": "still thinking", "logprobs": stream}
        )
        out = ollama(fake, "qwen3:1.7b").run("q")
        assert out.meta["has_logprobs"] is False
        assert out.meta["logprobs_unavailable_reason"] == "thinking_unaligned"

    def test_tagless_thinking_aligned_on_visible_text(self):
        stream = [lp("analysis", 0.4), lp(" stuff", 0.4), lp("Aus", 0.8), lp("tin", 0.95)]
        fake = FakeOllama(
            generate={"response": "Austin", "thinking": "analysis stuff", "logprobs": stream}
        )
        out = ollama(fake, "gpt-oss:20b").run("q")
        assert out.meta["has_logprobs"] is True
        assert out.top_tokens[0] == ("Aus", pytest.approx(0.8))
        mismatch = FakeOllama(generate={"response": "Paris", "thinking": "x", "logprobs": stream})
        out2 = ollama(mismatch, "gpt-oss:20b").run("q")
        assert out2.meta["has_logprobs"] is False
        assert out2.meta["logprobs_unavailable_reason"] == "thinking_unaligned"

    def test_close_tag_with_no_answer_tokens(self):
        stream = [lp("<think>", 0.9), lp("x", 0.5), lp("</think>", 0.9)]
        fake = FakeOllama(generate={"response": "", "thinking": "x", "logprobs": stream})
        out = ollama(fake, "qwen3:1.7b").run("q")
        assert out.meta["logprobs_unavailable_reason"] == "no_visible_tokens"


class TestOllamaRobustness:
    def test_transient_errors_are_retried(self):
        fake = FakeOllama()
        fake.fail = [503, 429]
        p = ollama(fake, max_retries=2)
        out = p.run("q")
        assert out.meta["completion"] == "Paris"
        assert out.meta["retries"] == 2 and p.last_retries == 2
        assert len(fake.generate_bodies()) == 3

    def test_retries_exhausted_raise_with_server_message(self):
        fake = FakeOllama()
        fake.fail = [503, 503]
        with pytest.raises(httpx.HTTPStatusError, match="busy"):
            ollama(fake, max_retries=1).run("q")

    def test_client_errors_are_not_retried(self):
        fake = FakeOllama()
        fake.fail = [404]
        with pytest.raises(httpx.HTTPStatusError, match="HTTP 404"):
            ollama(fake, max_retries=3).run("q")
        assert len(fake.generate_bodies()) == 1

    def test_connection_errors_are_retried_then_raised(self):
        attempts: list[int] = []

        def refuse(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            raise httpx.ConnectError("refused", request=request)

        p = OllamaProvider(
            model="m",
            base_url="http://x",
            transport=httpx.MockTransport(refuse),
            max_retries=2,
            retry_backoff=0.0,
        )
        with pytest.raises(httpx.ConnectError):
            p.run("q")
        assert len(attempts) == 3
        # Capability lookups never raise.
        assert p.server_version() is None and p.show() is None and p.capabilities() is None

    def test_usage_and_meta(self):
        fake = FakeOllama(
            generate={
                "response": "Paris",
                "prompt_eval_count": 12,
                "eval_count": 2,
                "done_reason": "stop",
            }
        )
        c = complete(ollama(fake), "q")
        assert (c.prompt_tokens, c.completion_tokens) == (12, 2)
        assert c.stop_reason == "stop" and c.framing == "chat"
        raw = ollama(fake).run("q", raw=True)
        assert raw.meta["framing"] == "raw"

    def test_benchmark_records_server_info(self):
        fake = FakeOllama(
            caps=["completion", "thinking"],
            generate={
                "response": "<think>x</think>Austin",
                "logprobs": [
                    lp("<think>", 0.9),
                    lp("x", 0.5),
                    lp("</think>", 0.9),
                    lp("Austin", 0.8),
                ],
            },
        )
        spec = ModelSpec("ollama", "qwen3:1.7b", factory=lambda: ollama(fake, "qwen3:1.7b"))
        res = run_benchmark([spec], ["multi_hop"], **FAST)
        assert res.validate() == []
        model = res.models[0]
        assert model["server_info"]["version"] == "0.34.4"
        assert model["server_info"]["thinking_model"] is True
        cell = res.cells[0]
        assert cell["passed"] is True
        assert cell["metrics"]["first_token_confidence"]["value"]["mean"] == pytest.approx(0.8)
        assert all(b.get("think") is False for b in fake.generate_bodies())
        md = render_markdown(res)
        assert "Ollama 0.34.4" in md and "thinking model: yes" in md
