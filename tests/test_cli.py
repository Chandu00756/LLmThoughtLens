"""End-to-end tests for the ``LLmThoughtLens`` command-line interface.

Every command is driven through :func:`LLmThoughtLens.cli.main` with an argv
list, using the deterministic mock provider and writing artefacts into
``tmp_path``.  Nothing touches the network, the user's config directory, a real
web server or a real terminal UI: ``run_server`` / ``run_tui`` are
monkeypatched and the config home is redirected to ``tmp_path``.

Stream contract: results go to stdout, status chatter (files written,
progress, warnings, input errors) to stderr — so ``trace --json`` stdout is
always parseable.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pytest
from LLmThoughtLens import __version__, cli
from LLmThoughtLens.providers.mock_provider import MockProvider
from rich.console import Console


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Wide console (no wrapping), isolated config home, CPU-only torch."""
    monkeypatch.setattr(cli, "_console", Console(width=400, color_system=None))
    monkeypatch.setattr(cli, "_err_console", Console(width=400, color_system=None, stderr=True))
    home = tmp_path / "home"
    monkeypatch.setenv("LLMTHOUGHTLENS_HOME", str(home))
    import LLmThoughtLens.tui.config as tui_config

    monkeypatch.setattr(tui_config, "CONFIG_DIR", home)
    monkeypatch.setattr(tui_config, "CONFIG_PATH", home / "config.json")
    try:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:  # pragma: no cover — torch-less install
        pass


def _run(capsys, argv: list[str]) -> tuple[int, str]:
    code = cli.main(argv)
    return code, capsys.readouterr().out


def _run_both(capsys, argv: list[str]) -> tuple[int, str, str]:
    """Like :func:`_run` but also return stderr (status chatter)."""
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# Dispatch / help / version / providers
# ---------------------------------------------------------------------------


class TestDispatch:
    @pytest.mark.parametrize("argv", [[], ["-h"], ["--help"]])
    def test_help_lists_every_command(self, capsys, argv):
        code, out = _run(capsys, argv)
        assert code == 0
        assert "Usage: LLmThoughtLens <command>" in out
        for name in cli._COMMANDS:
            assert f"  {name}" in out

    def test_unknown_command_returns_2(self, capsys):
        code, out = _run(capsys, ["frobnicate"])
        assert code == 2
        assert "unknown command: frobnicate" in out
        assert "Usage:" in out

    def test_main_reads_sys_argv_when_none(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["LLmThoughtLens", "version"])
        code, out = _run(capsys, None)  # type: ignore[arg-type]
        assert code == 0
        assert out.strip() == f"LLmThoughtLens {__version__}"

    def test_version(self, capsys):
        code, out = _run(capsys, ["version"])
        assert (code, out.strip()) == (0, f"LLmThoughtLens {__version__}")

    def test_providers_table(self, capsys):
        from LLmThoughtLens.providers.registry import available_providers, list_providers

        code, out = _run(capsys, ["providers"])
        assert code == 0
        avail = set(available_providers())
        for name in list_providers():
            line = next(ln for ln in out.splitlines() if f" {name} " in ln)
            assert ("yes" in line) == (name in avail)
        assert "mock" in avail

    def test_subcommand_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["trace", "--help"])
        assert exc.value.code == 0
        assert "--top-k" in capsys.readouterr().out

    def test_invalid_provider_choice_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["trace", "hi", "--provider", "not-a-provider"])
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _make_provider
# ---------------------------------------------------------------------------


class TestMakeProvider:
    def test_mock(self):
        p = cli._make_provider("mock", "", None, None)
        assert isinstance(p, MockProvider)

    def test_explicit_model_key_and_url_are_forwarded(self):
        pytest.importorskip("openai")
        p = cli._make_provider("openai", "my-model", "sk-explicit", None)
        assert p.model == "my-model"
        assert p._api_key == "sk-explicit"

    def test_empty_model_falls_back_to_a_default_and_env_key(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        p = cli._make_provider("openai", "", None, None)
        assert isinstance(p.model, str) and p.model  # some non-empty default
        assert p._api_key == "sk-env"

    def test_anthropic_env_key(self, monkeypatch):
        pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        p = cli._make_provider("anthropic", "claude-x", None, None)
        assert p.model == "claude-x"
        assert p._api_key == "sk-ant-env"

    def test_ollama_base_url(self):
        p = cli._make_provider("ollama", "tiny", None, "http://gpu-box:11434/")
        assert p.model == "tiny"
        assert p.base_url == "http://gpu-box:11434"

    def test_huggingface_is_lazy(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        p = cli._make_provider("huggingface", "some/local-model", None, None)
        assert p.model_name == "some/local-model"
        assert p._model is None  # nothing loaded / downloaded at construction

    def test_unknown_provider_raises(self):
        with pytest.raises(KeyError, match="unknown provider"):
            cli._make_provider("nope", "", None, None)


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------

PROMPT = "the capital of the state containing Dallas is"


class TestTrace:
    def test_json_summary_matches_direct_scope_run(self, capsys):
        from LLmThoughtLens.scope import Scope

        code, out = _run(capsys, ["trace", PROMPT, "--json", "--top-k", "5", "--threshold", "0.1"])
        assert code == 0
        summary = json.loads(out)

        ref = Scope(MockProvider(), top_k_features=5, attribution_threshold=0.1).trace_full(PROMPT)
        assert summary["prompt"] == PROMPT
        assert summary["output_token"] == ref.output_token
        assert summary["evidence_kind"] == "white_box"
        assert summary["n_features"] == len(ref.features)
        assert summary["n_graph_nodes"] == ref.graph.num_nodes
        assert summary["n_graph_edges"] == ref.graph.num_edges
        assert [t for t, _ in summary["top_tokens"]] == [t for t, _ in ref.top_tokens]
        assert summary["probes"] == []

    def test_json_with_probes(self, capsys):
        code, out = _run(capsys, ["trace", PROMPT, "--json", "--probes"])
        assert code == 0
        probes = json.loads(out)["probes"]
        assert len(probes) == 10
        assert all({"probe_name", "score", "passed"} <= set(p) for p in probes)
        assert all(0.0 <= p["score"] <= 1.0 for p in probes)

    def test_human_readable_output(self, capsys):
        code, out = _run(capsys, ["trace", PROMPT, "--probes"])
        assert code == 0
        assert f"provider  {MockProvider().model_id}" in out
        assert "evidence  white_box" in out
        assert "features" in out and "nodes," in out and "edges" in out
        assert "/ 10 passed" in out

    def test_output_writes_html_report(self, capsys, tmp_path):
        target = tmp_path / "report.html"
        code, out, err = _run_both(capsys, ["trace", PROMPT, "--output", str(target)])
        assert code == 0
        assert target.exists()
        html = target.read_text()
        assert html.lstrip().lower().startswith("<!doctype html")
        assert f"wrote HTML report to {target}" in err  # status chatter -> stderr
        assert "wrote HTML report" not in out
        assert "evidence  white_box" in out  # the human summary stays on stdout

    def test_json_stdout_stays_parseable_with_output(self, capsys, tmp_path):
        """Regression: 'wrote HTML report to …' was printed to stdout before the JSON."""
        target = tmp_path / "r.html"
        code, out, err = _run_both(capsys, ["trace", PROMPT, "--json", "--output", str(target)])
        assert code == 0
        summary = json.loads(out)  # the whole of stdout is one JSON document
        assert summary["prompt"] == PROMPT
        assert target.exists()
        assert f"wrote HTML report to {target}" in err

    def test_json_with_probes_and_output_stays_parseable(self, capsys, tmp_path):
        code, out, _ = _run_both(
            capsys, ["trace", PROMPT, "--json", "--probes", "--output", str(tmp_path / "r.html")]
        )
        assert code == 0
        assert len(json.loads(out)["probes"]) == 10


# ---------------------------------------------------------------------------
# probe / benchmark
# ---------------------------------------------------------------------------


class TestProbeAndBenchmark:
    def test_probe_prints_scorecard_and_writes_json(self, capsys, tmp_path):
        target = tmp_path / "scorecard.json"
        code, out, err = _run_both(capsys, ["probe", "--output", str(target)])
        assert code == 0
        data = json.loads(target.read_text())
        assert data["n_total"] == 10
        assert data["provider"]
        assert len(data["results"]) == 10
        assert f"overall  {data['n_passed']} / 10 passed" in out
        for r in data["results"]:
            assert r["probe_name"] in out
        assert f"wrote scorecard to {target}" in err
        assert "wrote scorecard" not in out

    def test_probe_without_output_writes_nothing(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, out = _run(capsys, ["probe"])
        assert code == 0
        assert "overall" in out
        assert not any(p.suffix == ".json" for p in tmp_path.rglob("*"))

    def test_benchmark_matches_probe_battery(self, capsys, tmp_path):
        from LLmThoughtLens.probes.builtin import all_probes
        from LLmThoughtLens.probes.runner import ProbeRunner

        target = tmp_path / "bench.json"
        code, out, err = _run_both(capsys, ["benchmark", "--output", str(target)])
        assert code == 0
        data = json.loads(target.read_text())
        ref = ProbeRunner(all_probes()).run_all(MockProvider())
        assert data["n_total"] == ref.n_total == 10
        assert data["n_passed"] == ref.n_passed
        assert data["mean_score"] == pytest.approx(ref.mean_score)
        assert "Interpretability scorecard" in out
        assert f"provider={MockProvider().model_id}" in out
        assert f"wrote {target.resolve()}" in err
        assert str(target.resolve()) not in out

    def test_benchmark_default_output_path(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, _ = _run(capsys, ["benchmark"])
        assert code == 0
        assert json.loads((tmp_path / "benchmark_results.json").read_text())["n_total"] == 10


# ---------------------------------------------------------------------------
# serve / tui (launchers are monkeypatched — nothing actually starts)
# ---------------------------------------------------------------------------


class TestLaunchers:
    def test_serve_passes_host_port_and_browser_flag(self, capsys, monkeypatch):
        pytest.importorskip("fastapi")
        import LLmThoughtLens.server.app as server_app

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(server_app, "run_server", lambda **kw: calls.append(kw))
        code, out = _run(capsys, ["serve", "--host", "0.0.0.0", "--port", "9123", "--no-open"])
        assert code == 0
        assert calls == [{"host": "0.0.0.0", "port": 9123, "open_browser": False}]
        assert "http://0.0.0.0:9123/" in out
        assert "http://0.0.0.0:9123/v1" in out

    def test_serve_defaults_open_browser(self, capsys, monkeypatch):
        pytest.importorskip("fastapi")
        import LLmThoughtLens.server.app as server_app

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(server_app, "run_server", lambda **kw: calls.append(kw))
        assert cli.main(["serve"]) == 0
        assert calls[0]["open_browser"] is True
        assert calls[0]["host"] == "127.0.0.1"
        assert isinstance(calls[0]["port"], int)

    def test_serve_without_server_extra_returns_1(self, capsys, monkeypatch):
        monkeypatch.setitem(sys.modules, "LLmThoughtLens.server.app", None)
        code, out = _run(capsys, ["serve"])
        assert code == 1
        assert "needs the 'server' extra" in out

    def test_tui_launches_run_tui(self, monkeypatch):
        pytest.importorskip("textual")
        import LLmThoughtLens.tui.app as tui_app

        launched: list[bool] = []
        monkeypatch.setattr(tui_app, "run_tui", lambda *a, **k: launched.append(True))
        assert cli.main(["tui"]) == 0
        assert launched == [True]


# ---------------------------------------------------------------------------
# SAE pipeline: cache-activations -> train-sae -> label-features
# ---------------------------------------------------------------------------

CORPUS = [
    "the cat sat on the mat",
    "dogs chase the cat",
    "paris is the capital of france",
    "austin is the capital of texas",
]


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(CORPUS) + "\n")
    return path


class TestSAEPipeline:
    def test_cache_activations_with_mock(self, capsys, tmp_path, corpus_file):
        out_path = tmp_path / "acts.npz"
        code, out, err = _run_both(
            capsys,
            [
                "cache-activations",
                "--provider",
                "mock",
                "--corpus",
                str(corpus_file),
                "--layer",
                "2",
                "--output",
                str(out_path),
            ],
        )
        assert code == 0
        from LLmThoughtLens.features.cache import ActivationCache

        data = ActivationCache.load(out_path)
        mock = MockProvider()
        expected = np.concatenate([mock.run(p).activations[2] for p in CORPUS], axis=0)
        np.testing.assert_array_equal(data["activations"], expected)
        assert data["meta"]["layer"] == 2
        assert data["meta"]["provider"] == "mock"
        n_tok = sum(len(p.split()) for p in CORPUS)
        assert f"saved {n_tok} tokens × {mock.d_model}" in err
        assert "prompt 1:" in err  # verbose progress is chatter -> stderr
        assert out == ""
        # The provider's own tokens and the row -> (line, position) map are stored.
        assert data["tokens"] == [t for p in CORPUS for t in p.split()]
        assert data["prompts"] == CORPUS
        assert data["prompt_lines"].tolist() == [1, 2, 3, 4]
        streams, positions = ActivationCache.token_streams(data)
        assert streams == [p.split() for p in CORPUS]
        assert positions[:3] == [(0, 0), (0, 1), (0, 2)]

    def test_cache_activations_respects_max_tokens(self, capsys, tmp_path, corpus_file):
        out_path = tmp_path / "acts.npz"
        cli.main(
            [
                "cache-activations",
                "--provider",
                "mock",
                "--corpus",
                str(corpus_file),
                "--max-tokens",
                "8",
                "--output",
                str(out_path),
            ]
        )
        from LLmThoughtLens.features.cache import ActivationCache

        assert ActivationCache.load(out_path)["activations"].shape[0] == 8

    def test_train_sae_then_label_features(self, capsys, tmp_path, corpus_file):
        pytest.importorskip("torch")
        from LLmThoughtLens.features.sae import SparseAutoencoder

        acts = tmp_path / "acts.npz"
        sae_path = tmp_path / "sae.pt"
        labelled_path = tmp_path / "sae_labelled.pt"

        assert (
            cli.main(
                [
                    "cache-activations",
                    "--provider",
                    "mock",
                    "--corpus",
                    str(corpus_file),
                    "--output",
                    str(acts),
                ]
            )
            == 0
        )
        capsys.readouterr()

        code, out, err = _run_both(
            capsys,
            [
                "train-sae",
                "--activations",
                str(acts),
                "--output",
                str(sae_path),
                "--dict-size",
                "32",
                "--k",
                "4",
                "--steps",
                "6",
                "--batch-size",
                "8",
                "--lr",
                "1e-3",
            ],
        )
        assert code == 0
        sae = SparseAutoencoder.load(sae_path)
        assert sae.config.input_dim == MockProvider().d_model
        assert sae.config.dict_size == 32
        assert sae.config.k == 4
        assert sae._trained is True and sae._steps_run == 6
        n_tok = sum(len(line.split()) for line in CORPUS)
        assert f"training SAE on {n_tok} × {MockProvider().d_model} activations" in err
        assert "l0_mean=" in out and "dead=" in out and "mse=" in out  # the result
        assert f"saved SAE to {sae_path}" in err

        code, out, err = _run_both(
            capsys,
            [
                "label-features",
                "--sae",
                str(sae_path),
                "--activations",
                str(acts),
                "--corpus",
                str(corpus_file),
                "--labeler-provider",
                "mock",
                "--output",
                str(labelled_path),
            ],
        )
        assert code == 0
        labelled = SparseAutoencoder.load(labelled_path)
        from LLmThoughtLens.features.cache import ActivationCache

        codes = labelled.encode(ActivationCache.load(acts)["activations"])
        live = {int(i) for i in np.where((codes != 0).any(axis=0))[0]}
        assert set(labelled.labels) == live
        assert all(isinstance(v, str) and v for v in labelled.labels.values())
        assert f"labelled {len(live)} features" in err
        assert "--corpus is ignored" in err  # stored provider tokens win


# ---------------------------------------------------------------------------
# label-features alignment: cached activations <-> provider tokens
# ---------------------------------------------------------------------------


class _SubwordMock(MockProvider):
    """Mock with a BPE-like tokenizer: every word is split into 2-char pieces.

    Its activation rows therefore outnumber the corpus's whitespace words —
    exactly the HuggingFace situation that misaligned ``label-features``.
    """

    def run(self, prompt: str, **kwargs: Any):
        pieces = [w[i : i + 2] for w in prompt.split() for i in range(0, len(w), 2)]
        out = super().run(" ".join(pieces), **kwargs)
        out.prompt = prompt
        return out


class _StubSAE:
    """SAE stand-in with scripted codes (no torch); records what gets saved."""

    def __init__(self, codes: np.ndarray) -> None:
        self.codes = np.asarray(codes, dtype=np.float32)
        self._labels: dict[int, str] = {}

    def encode(self, activations: np.ndarray) -> np.ndarray:
        assert activations.shape[0] == self.codes.shape[0]
        return self.codes

    def set_label(self, feature_id: int, label: str) -> None:
        self._labels[int(feature_id)] = str(label)

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._labels)

    def save_with_labels(self, path, labels: dict[int, str]) -> None:
        self._labels.update(labels)
        with open(path, "w") as fh:
            json.dump({str(k): v for k, v in self._labels.items()}, fh)


class _RecordingLabeler(MockProvider):
    """Black-box-style labelling LLM: fixed completion, records every prompt."""

    evidence_kind = "black_box"

    def __init__(self, completion: str = "Capital Cities") -> None:
        super().__init__()
        self.completion = completion
        self.prompts: list[str] = []

    def run(self, prompt: str, **kwargs: Any):
        from LLmThoughtLens.providers.base import ProviderOutput

        self.prompts.append(prompt)
        return ProviderOutput(
            prompt=prompt, tokens=self.completion.split(), meta={"completion": self.completion}
        )


def _snippets(prompt: str) -> list[str]:
    return [ln[2:] for ln in prompt.splitlines() if ln.startswith("- ")]


@pytest.fixture
def labelling(monkeypatch):
    """Route cache providers / labeler / SAE loading through test doubles.

    Returns a dict the test fills: ``cache_provider`` (used by
    cache-activations), ``codes`` (the SAE's scripted codes) and, after the
    run, ``labeler`` (the recording labelling provider).
    """
    from LLmThoughtLens.features.sae import SparseAutoencoder

    state: dict[str, Any] = {"cache_provider": MockProvider(), "codes": None}

    def fake_make_provider(provider, model, api_key, base_url):
        if provider == "labeler":
            state["labeler"] = _RecordingLabeler()
            return state["labeler"]
        return state["cache_provider"]

    monkeypatch.setattr(cli, "_make_provider", fake_make_provider)
    monkeypatch.setattr(
        SparseAutoencoder, "load", classmethod(lambda cls, path: _StubSAE(state["codes"]))
    )
    return state


def _cache(capsys, corpus_path, out_path) -> dict[str, Any]:
    from LLmThoughtLens.features.cache import ActivationCache

    code = cli.main(["cache-activations", "--corpus", str(corpus_path), "--output", str(out_path)])
    assert code == 0
    capsys.readouterr()
    return ActivationCache.load(out_path)


def _label(capsys, acts, tmp_path, *extra: str) -> tuple[int, str, str]:
    return _run_both(
        capsys,
        [
            "label-features",
            "--sae",
            str(tmp_path / "sae.pt"),
            "--activations",
            str(acts),
            "--labeler-provider",
            "labeler",
            "--output",
            str(tmp_path / "labels.json"),
            *extra,
        ],
    )


def _one_hot(n_rows: int, row: int) -> np.ndarray:
    codes = np.zeros((n_rows, 2), dtype=np.float32)
    codes[row, 0] = 1.0
    return codes


class TestLabelFeaturesAlignment:
    def test_cache_skips_blank_lines_and_records_corpus_lines(self, capsys, tmp_path, labelling):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the cat sat\n\n   \nparis is the capital of france\n")
        data = _cache(capsys, corpus, tmp_path / "acts.npz")
        assert data["prompts"] == ["the cat sat", "paris is the capital of france"]
        assert data["prompt_lines"].tolist() == [1, 4]
        # No phantom "<empty>" rows for blank lines.
        assert data["activations"].shape[0] == 9
        assert data["tokens"] == [
            "the",
            "cat",
            "sat",
            "paris",
            "is",
            "the",
            "capital",
            "of",
            "france",
        ]
        assert data["meta"]["tokens_aligned"] is True

    def test_blank_lines_no_longer_shift_contexts(self, capsys, tmp_path, labelling):
        """Regression: blank lines cached a row but were skipped when labelling."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("\nthe cat sat\n\nparis is the capital of france\n\n")
        acts = tmp_path / "acts.npz"
        data = _cache(capsys, corpus, acts)
        row = data["tokens"].index("france")
        labelling["codes"] = _one_hot(data["activations"].shape[0], row)

        code, out, err = _label(capsys, acts, tmp_path)
        assert code == 0, err
        (prompt,) = labelling["labeler"].prompts
        assert _snippets(prompt) == ["paris is the capital of «france»"]
        assert json.loads((tmp_path / "labels.json").read_text()) == {"0": "capital cities"}
        assert "feature     0 → capital cities" in err  # per-feature progress on stderr

    def test_subword_tokens_are_taken_from_the_cache_not_the_corpus(
        self, capsys, tmp_path, labelling
    ):
        """BPE-style providers: contexts come from the provider's own pieces."""
        labelling["cache_provider"] = _SubwordMock()
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("austin is the capital of texas\n")
        acts = tmp_path / "acts.npz"
        data = _cache(capsys, corpus, acts)
        assert data["tokens"] == ["au", "st", "in", "is", "th", "e", "ca", "pi", "ta", "l"] + [
            "of",
            "te",
            "xa",
            "s",
        ]
        row = data["tokens"].index("xa")
        labelling["codes"] = _one_hot(len(data["tokens"]), row)

        code, _, err = _label(capsys, acts, tmp_path, "--corpus", str(corpus))
        assert code == 0, err
        (prompt,) = labelling["labeler"].prompts
        # 8-token window of the provider's pieces, not the corpus words.
        assert _snippets(prompt) == ["th e ca pi ta l of te «xa» s"]
        assert "--corpus is ignored" in err

    def test_legacy_cache_with_matching_whitespace_corpus_warns(self, capsys, tmp_path, labelling):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the cat sat\n\ndogs chase cats\n")
        acts = tmp_path / "legacy.npz"
        activations = np.concatenate(
            [MockProvider().run(p).activations[0] for p in ("the cat sat", "dogs chase cats")]
        )
        np.savez_compressed(acts, activations=activations, meta__provider="mock")
        labelling["codes"] = _one_hot(6, 4)

        code, _, err = _label(capsys, acts, tmp_path, "--corpus", str(corpus))
        assert code == 0
        assert "predates token storage" in err and "not guaranteed" in err
        (prompt,) = labelling["labeler"].prompts
        assert _snippets(prompt) == ["dogs «chase» cats"]  # contexts stay within a line

    def test_legacy_cache_with_mismatched_corpus_is_refused(self, capsys, tmp_path, labelling):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("austin is the capital of texas\n")
        acts = tmp_path / "legacy.npz"
        sub = _SubwordMock().run("austin is the capital of texas").activations[0]
        np.savez_compressed(acts, activations=sub, meta__provider="huggingface")

        code, out, err = _label(capsys, acts, tmp_path, "--corpus", str(corpus))
        assert code == 2
        assert "6 whitespace tokens" in err and f"{sub.shape[0]} activation rows" in err
        assert "can't be aligned" in err and "cache-activations" in err
        assert out == ""
        assert not (tmp_path / "labels.json").exists()
        assert "labeler" not in labelling  # nothing was sent to the labelling LLM

    def test_legacy_cache_without_corpus_is_refused(self, capsys, tmp_path, labelling):
        acts = tmp_path / "legacy.npz"
        np.savez_compressed(acts, activations=np.zeros((3, 4), dtype=np.float32))
        code, _, err = _label(capsys, acts, tmp_path)
        assert code == 2
        assert "predates token storage" in err

    def test_corrupt_token_mapping_is_refused(self, capsys, tmp_path, labelling):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("a b c\n")
        acts = tmp_path / "acts.npz"
        _cache(capsys, corpus, acts)
        with np.load(acts) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["token_pos"] = np.array([0, 2, 1])
        np.savez_compressed(acts, **arrays)
        code, _, err = _label(capsys, acts, tmp_path)
        assert code == 2
        assert "corrupt activation cache" in err

    def test_tiny_huggingface_end_to_end_uses_provider_tokens(self, capsys, tmp_path, labelling):
        """Real HuggingFaceProvider (offline tiny GPT-2): contexts use *its* token strings."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from _tiny_hf import WordTokenizer, make_provider

        labelling["cache_provider"] = make_provider()
        lines = ["the cat sat", "", "paris is the capital of france"]
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("\n".join(lines) + "\n")
        acts = tmp_path / "acts.npz"
        data = _cache(capsys, corpus, acts)

        tok = WordTokenizer()
        expected = [tok.decode([tok.word_id(w)]) for line in lines for w in line.split()]
        assert data["tokens"] == expected  # "<id>" pieces, not the corpus words
        assert data["activations"].shape == (len(expected), 32)
        assert data["meta"]["provider"] == "huggingface"
        assert data["prompt_lines"].tolist() == [1, 3]

        row = 3 + 5  # "france": last token of the second (non-blank) prompt
        labelling["codes"] = _one_hot(len(expected), row)
        code, _, err = _label(capsys, acts, tmp_path, "--corpus", str(corpus))
        assert code == 0, err
        (prompt,) = labelling["labeler"].prompts
        assert _snippets(prompt) == [" ".join(expected[3:8]) + f" «{expected[8]}»"]
