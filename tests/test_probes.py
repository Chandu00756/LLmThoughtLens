"""Tests for the 10 built-in probes + the probe runner."""

from __future__ import annotations

from LLmThoughtLens.probes.base import BaseProbe, ProbeResult
from LLmThoughtLens.probes.builtin import (
    BUILTIN_PROBES,
    MultiHopProbe,
    all_probes,
    probe_by_name,
)
from LLmThoughtLens.probes.runner import ProbeReport, ProbeRunner
from LLmThoughtLens.providers.mock_provider import MockProvider


class TestRegistry:
    def test_ten_probes_registered(self):
        assert len(BUILTIN_PROBES) == 10

    def test_each_probe_has_unique_name(self):
        names = [cls.name for cls in BUILTIN_PROBES]
        assert len(set(names)) == len(names)

    def test_probe_by_name_roundtrip(self):
        for cls in BUILTIN_PROBES:
            inst = probe_by_name(cls.name)
            assert isinstance(inst, cls)

    def test_all_probes_instantiates_one_of_each(self):
        probes = all_probes()
        assert len(probes) == 10
        assert all(isinstance(p, BaseProbe) for p in probes)


class TestProbeContract:
    def test_each_probe_returns_probe_result(self):
        mp = MockProvider(seed=1)
        for probe in all_probes():
            res = probe.run(mp)
            assert isinstance(res, ProbeResult)
            assert 0.0 <= res.score <= 1.0
            assert isinstance(res.passed, bool)
            assert isinstance(res.evidence, dict)
            assert res.probe_name == probe.name

    def test_probe_result_is_json_safe(self):
        mp = MockProvider(seed=1)
        res = MultiHopProbe().run(mp)
        import json

        assert json.dumps(res.as_dict(), default=str)


class TestRunner:
    def test_runner_produces_report(self):
        runner = ProbeRunner(all_probes())
        report = runner.run_all(MockProvider(seed=4))
        assert isinstance(report, ProbeReport)
        assert report.n_total == 10
        assert 0.0 <= report.mean_score <= 1.0
        assert all(isinstance(r, ProbeResult) for r in report.results)

    def test_runner_progress_callback(self):
        seen: list[str] = []

        def cb(i, n, probe):
            seen.append(probe.name)

        ProbeRunner(all_probes()).run_all(MockProvider(seed=4), progress_callback=cb)
        assert len(seen) == 10

    def test_runner_handles_exceptions(self):
        class Boom(BaseProbe):
            name = "boom"

            def run(self, provider, prompt=None):
                raise RuntimeError("explode")

        report = ProbeRunner([Boom()]).run_all(MockProvider(seed=0))
        assert report.results[0].passed is False
        assert "explode" in report.results[0].summary
