"""Surfaces extract.attempt.digest and distill.evidence (KHSI)."""

import json

import pytest

import ksi.distillation.distiller as distiller
import ksi.distillation.per_task as per_task
import ksi.distillation.prompts as prompts
import ksi.orchestrator.enrichment_phase as enrichment_phase
import ksi.orchestrator.execution_phase as execution_phase
from ksi import harness_surfaces
from ksi.distillation.types import DistillInput
from ksi.memory.knowledge_store import KnowledgeStore
from ksi.models import TaskTrace

HARNESS = """
def digest(default, attempt):
    calls = [c.get("tool_name") for c in attempt["tool_trace"]]
    return f"task {attempt['task_id']} calls={calls}\\nline two"

SURFACES = {
    "extract.attempt.digest": digest,
    "distill.evidence": lambda default: {"cite_attempts": True},
}
"""


@pytest.fixture(autouse=True)
def _no_harness(monkeypatch):
    monkeypatch.delenv(harness_surfaces.ENV_VAR, raising=False)
    harness_surfaces.install(None)
    yield
    harness_surfaces.install(None)


def _trace(**kw):
    return TaskTrace(
        generation=1,
        agent_id="agent-0",
        task_id="t1",
        model_output="done",
        native_score=0.0,
        tool_trace=[{"type": "tool_call", "tool_name": "shell", "tool_output": "ok"}],
        **kw,
    )


def _record(store, digest):
    return store.record_attempt(
        task_id="t1", agent_id="agent-0", generation=1, model_output="done", native_score=0.0, trace_digest=digest
    )


def test_defaults_keep_nothing_and_store_unchanged(tmp_path):
    assert execution_phase.trace_digest_for(_trace()) == ""
    store = KnowledgeStore(str(tmp_path / "k.sqlite"))
    _record(store, "")
    page = store.query_task("t1", generation=None, entry_types=["attempt"])
    attempt = page["attempts"][0]
    assert "trace_digest" not in attempt["content"]
    assert isinstance(attempt["id"], int)
    inp = DistillInput(generation=1, task_ids=["t1"], knowledge_store=store, llm=None)
    loaded = distiller._load_attempts(inp, "t1")
    assert "trace_digest" not in loaded[0]
    assert "trace_digest" not in enrichment_phase._knowledge_attempts_to_seed_records(page)[0]
    system, user = prompts.build_per_task_distill_prompt(task_id="t1", attempts=loaded, posts=[])
    assert prompts._VERBATIM_FORUM_RULE in system and "attempt_id=" not in user
    assert prompts.evidence_policy() == {"cite_attempts": False}


def test_digest_flows_to_store_distill_and_seed(tmp_path):
    harness = tmp_path / "h.py"
    harness.write_text(HARNESS)
    harness_surfaces.install(harness)
    digest = execution_phase.trace_digest_for(_trace())
    assert digest == "task t1 calls=['shell']\nline two"
    assert execution_phase.trace_digest_for(_trace(error="boom")) == ""

    store = KnowledgeStore(str(tmp_path / "k.sqlite"))
    attempt_id = _record(store, digest)
    page = store.query_task("t1", generation=None, entry_types=["attempt"])
    assert page["attempts"][0]["content"]["trace_digest"] == digest
    assert enrichment_phase._knowledge_attempts_to_seed_records(page)[0]["trace_digest"] == digest

    inp = DistillInput(generation=1, task_ids=["t1"], knowledge_store=store, llm=None)
    loaded = distiller._load_attempts(inp, "t1")
    assert loaded[0]["trace_digest"] == digest and loaded[0]["id"] == attempt_id
    system, user = prompts.build_per_task_distill_prompt(task_id="t1", attempts=loaded, posts=[])
    assert prompts._VERBATIM_ATTEMPT_RULE in system and prompts._VERBATIM_FORUM_RULE not in system
    assert prompts._CITE_POSTS_OR_ATTEMPTS_RULE in system
    assert f"attempt_id={attempt_id} " in user and "    line two" in user


def test_cited_attempt_is_valid_evidence_only_with_the_policy(tmp_path):
    attempts = [{"id": 41, "agent_id": "a", "generation": 1, "native_score": 0.0, "trace_digest": "x"}]
    item = {
        "text": "FALSIFIED: recolour each 4 to the nearer color-2 cell by Manhattan distance, ties to 6",
        "applies_when": "markers of two colours",
        "does_not_apply_when": "single marker",
        "evidence": [{"post_id": 41, "quote": "x"}],
    }
    payload = {"rejected_hypotheses": [item]}

    def llm(system, user, **kw):
        return json.dumps(payload)

    def evidence():
        bundle = per_task.distill_one_task(task_id="t1", attempts=attempts, posts=[], llm=llm)
        return bundle.rejected_hypotheses[0].get("evidence")

    assert evidence() is None
    harness = tmp_path / "h.py"
    harness.write_text(HARNESS)
    harness_surfaces.install(harness)
    assert evidence() == [{"post_id": 41, "quote": "x"}]
