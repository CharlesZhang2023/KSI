"""Tests for replaceable knowledge-harness surfaces and the KHSI hook strategy."""

import json
import sys
from unittest.mock import MagicMock

import pytest

import ksi.discussion.prompts  # noqa: F401 - registers surfaces
import ksi.distillation.cross_task  # noqa: F401
import ksi.distillation.distiller as distiller
import ksi.distillation.per_task as per_task
import ksi.distillation.prompts  # noqa: F401
import ksi.forum.prompt  # noqa: F401
import ksi.memory.cross_task_context as ctc
import ksi.orchestrator.enrichment_phase as enrichment_phase
import ksi.orchestrator.execution_phase as execution_phase
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


def test_every_surface_states_its_contract_and_replayability():
    for name, text in harness_surfaces.KNOWN_SURFACES.items():
        assert "fn(default" in text and "->" in text, name
        assert text.rstrip().endswith(("[offline]", "[online]")), name


def test_new_surfaces_default_to_the_original_behaviour(monkeypatch):
    attempts, posts = [{"generation": 1}], [{"id": 3}]
    assert distiller.select_per_task_inputs(attempts, posts, task_id="t", generation=2, knowledge_store=None) == (
        attempts,
        posts,
    )
    assert distiller.select_cross_task_inputs(posts, generation=2, knowledge_store=None) is posts
    prior, related = [{"a": 1}], [{"task_id": "u"}]
    assert enrichment_phase.select_seed_context(prior, related, task_id="t", task_prompt="p", all_summaries=[]) == (
        prior,
        related,
    )

    monkeypatch.delenv("KSI_PER_TASK_DISTILL_GEN_WINDOW", raising=False)
    monkeypatch.setenv("KSI_CROSS_TASK_DISTILL_GEN_WINDOW", "3")
    assert distiller._per_task_distill_gen_window() == 6
    assert distiller._cross_task_distill_gen_window() == 3

    short = execution_phase.reflection_inputs("x" * 400, "a\nb")
    assert short == ("x" * 300, "a b")
    long_output = "h" * 1300 + "m" * 500 + "t" * 1300
    preview, excerpt = execution_phase.reflection_inputs("task", long_output)
    assert preview == "task"
    assert excerpt == "h" * 1200 + " ...(truncated)... " + "t" * 1200

    items = per_task._as_insight_list(
        [{"text": "rotate the grid 90 degrees", "evidence": [{"post_id": 9}]}],
        allowed_post_ids={1},
    )
    assert items == [{"text": "rotate the grid 90 degrees"}]


def test_new_surfaces_can_be_overridden(tmp_path):
    harness = tmp_path / "harness.py"
    harness.write_text(
        "def window(default):\n"
        "    return 0\n"
        "def inputs(default, attempts, posts, **kw):\n"
        "    return attempts[-1:], []\n"
        "def parse(default, value, **kw):\n"
        "    return default(value, **kw)[:1]\n"
        "def excerpt(default, task_prompt, model_output):\n"
        "    return task_prompt, model_output\n"
        "def context(default, prior, related, **kw):\n"
        "    return prior[:1], []\n"
        "SURFACES = {\n"
        "    'distill.per_task.window': window,\n"
        "    'distill.per_task.inputs': inputs,\n"
        "    'distill.parse_items': parse,\n"
        "    'extract.reflection.inputs': excerpt,\n"
        "    'seed.select_context': context,\n"
        "}\n"
    )
    harness_surfaces.install(harness)
    assert distiller._per_task_distill_gen_window() == 0
    assert distiller.select_per_task_inputs([1, 2], [3], task_id="t", generation=1, knowledge_store=None) == ([2], [])
    two = [{"text": "rotate the grid"}, {"text": "flip the grid"}]
    assert len(per_task._as_insight_list(two)) == 1
    assert execution_phase.reflection_inputs("full statement", "x" * 3000)[1] == "x" * 3000
    assert enrichment_phase.select_seed_context([1, 2], [3], task_id="t", task_prompt="", all_summaries=[]) == ([1], [])
    assert harness_surfaces.status()["call_errors"] == {}
