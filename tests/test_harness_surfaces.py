"""Tests for replaceable knowledge-harness surfaces and the KHSI hook strategy."""

import json
import sys
from unittest.mock import MagicMock

import pytest

import ksi.distillation.per_task  # noqa: F401 - registers surfaces
import ksi.distillation.prompts  # noqa: F401
import ksi.forum.prompt  # noqa: F401
import ksi.memory.cross_task_context as ctc
import ksi.runtime.seeding  # noqa: F401
from ksi import harness_surfaces
from ksi.models import GenerationConfig
from ksi.orchestrator.strategy import GenerationContext, KnowledgeHarnessHookStrategy, get_strategy_spec


@pytest.fixture(autouse=True)
def _no_harness(monkeypatch):
    monkeypatch.delenv(harness_surfaces.ENV_VAR, raising=False)
    harness_surfaces.install(None)
    yield
    harness_surfaces.install(None)


def test_every_known_surface_is_registered():
    assert set(harness_surfaces.DEFAULTS) == set(harness_surfaces.KNOWN_SURFACES)
    assert ctc.target_relevance.__ksi_surface__ == "retrieve.cross_task.relevance"


def test_default_path_is_the_original_function():
    tokens = frozenset({"grid", "color"})
    post = {"text": "grid rotate"}
    assert (
        ctc.target_relevance(post, tokens)
        == harness_surfaces.DEFAULTS["retrieve.cross_task.relevance"](post, tokens)
        == 0.5
    )
    assert harness_surfaces.status() == {
        "path": None,
        "digest": None,
        "overrides": [],
        "error": None,
        "call_errors": {},
    }


def test_override_wraps_default_and_reloads_on_change(tmp_path):
    harness = tmp_path / "harness.py"
    harness.write_text(
        "def rel(default, post, tokens):\n"
        "    return default(post, tokens) * 2\n"
        "SURFACES = {'retrieve.cross_task.relevance': rel}\n"
    )
    harness_surfaces.install(harness)
    tokens = frozenset({"grid", "color"})
    assert ctc.target_relevance({"text": "grid"}, tokens) == 1.0
    first = harness_surfaces.status()
    assert first["overrides"] == ["retrieve.cross_task.relevance"] and first["error"] is None

    harness.write_text("SURFACES = {}\n")
    assert ctc.target_relevance({"text": "grid"}, tokens) == 0.5
    assert harness_surfaces.status()["digest"] != first["digest"]


def test_env_var_selects_harness(tmp_path, monkeypatch):
    harness = tmp_path / "harness.py"
    harness.write_text("SURFACES = {'retrieve.cross_task.relevance': lambda d, p, t: 0.25}\n")
    monkeypatch.setenv(harness_surfaces.ENV_VAR, str(harness))
    assert ctc.target_relevance({"text": "x"}, frozenset({"y"})) == 0.25


@pytest.mark.parametrize(
    "body, error",
    [
        ("raise RuntimeError('boom')\n", "RuntimeError"),
        ("SURFACES = {'no.such.surface': print}\n", "KeyError"),
        ("SURFACES = []\n", "TypeError"),
    ],
)
def test_broken_harness_falls_back_to_defaults(tmp_path, body, error):
    harness = tmp_path / "harness.py"
    harness.write_text(body)
    harness_surfaces.install(harness)
    assert ctc.target_relevance({"text": "grid"}, frozenset({"grid", "color"})) == 0.5
    state = harness_surfaces.status()
    assert state["overrides"] == [] and state["error"].startswith(error)


def test_override_exception_falls_back_but_default_exception_propagates(tmp_path):
    harness = tmp_path / "harness.py"
    harness.write_text(
        "def rel(default, post, tokens):\n"
        "    if post.get('boom'):\n"
        "        raise ValueError('harness bug')\n"
        "    return default(post, tokens)\n"
        "SURFACES = {'retrieve.cross_task.relevance': rel}\n"
    )
    harness_surfaces.install(harness)
    tokens = frozenset({"grid", "color"})
    assert ctc.target_relevance({"text": "grid", "boom": True}, tokens) == 0.5
    assert harness_surfaces.status()["call_errors"] == {"retrieve.cross_task.relevance": 1}
    with pytest.raises(AttributeError):
        ctc.target_relevance(None, tokens)  # the original implementation fails on None


def test_khsi_strategy_is_registered():
    assert isinstance(get_strategy_spec("khsi").factory(), KnowledgeHarnessHookStrategy)


def _ctx(tmp_path, generation, num_generations=3):
    config = GenerationConfig(
        num_agents=1, num_generations=num_generations, knowledge_db_path=str(tmp_path / "k.sqlite")
    )
    return GenerationContext(generation=generation, fresh_traces=[], phases=MagicMock(), config=config)


def test_hook_rewrites_harness_before_forum(tmp_path, monkeypatch):
    harness = tmp_path / "harness.py"
    harness.write_text("SURFACES = {}\n")
    script = tmp_path / "hook.py"
    script.write_text(
        "import os, pathlib\n"
        "g = os.environ['KSI_KHSI_GENERATION']\n"
        "pathlib.Path(os.environ['KSI_KNOWLEDGE_HARNESS']).write_text(\n"
        "    'SURFACES = {\"retrieve.cross_task.relevance\": lambda d, p, t: ' + g + '}\\n')\n"
    )
    monkeypatch.setenv(harness_surfaces.ENV_VAR, str(harness))
    monkeypatch.setenv("KSI_KHSI_HOOK_CMD", f"{sys.executable} {script}")
    ctx = _ctx(tmp_path, generation=2)
    KnowledgeHarnessHookStrategy().per_task_forum(ctx)

    ctx.phases.per_task_forum.assert_called_once()
    assert ctc.target_relevance({"text": "x"}, frozenset({"y"})) == 2
    rows = [json.loads(line) for line in (tmp_path / "khsi_harness_versions.jsonl").read_text().splitlines()]
    assert rows[0]["generation"] == 2 and rows[0]["hook_rc"] == 0
    assert rows[0]["overrides"] == ["retrieve.cross_task.relevance"]


def test_hook_failure_keeps_running_and_final_generation_skips_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("KSI_KHSI_HOOK_CMD", f"{sys.executable} -c 'import sys; sys.exit(3)'")
    strategy = KnowledgeHarnessHookStrategy()
    strategy.per_task_forum(_ctx(tmp_path, generation=1))
    strategy.per_task_forum(_ctx(tmp_path, generation=3))
    rows = [json.loads(line) for line in (tmp_path / "khsi_harness_versions.jsonl").read_text().splitlines()]
    assert [r["hook_rc"] for r in rows] == [3, None]
