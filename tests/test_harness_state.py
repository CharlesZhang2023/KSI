"""Surface state.update: the harness's own state across generations (KHSI)."""

import sqlite3

import pytest

import ksi.distillation.distiller as distiller
from ksi import harness_state, harness_surfaces
from ksi.distillation.types import DistillInput
from ksi.memory.knowledge_store import CROSS_TASK_SENTINEL, KnowledgeStore

# A ledger harness: per bundle item text, how often it was shown and how often
# the task was then solved; plus the generations it has seen, in order.
LEDGER = """
def update(default, state, feedback):
    state = default(state, feedback)
    state.setdefault("generations", []).append(feedback["generation"])
    ledger = state.setdefault("ledger", {})
    for task_id, task in feedback["tasks"].items():
        for scope in ("per_task", "cross_task"):
            bundle = task["seeded"][scope] or {}
            for item in bundle.get("transferable_insights", []):
                row = ledger.setdefault(item["text"], {"shown": 0, "solved": 0})
                row["shown"] += 1
                row["solved"] += int(task["solved"])
    return state

SURFACES = {"state.update": update}
"""


@pytest.fixture(autouse=True)
def _no_harness(monkeypatch):
    monkeypatch.delenv(harness_surfaces.ENV_VAR, raising=False)
    harness_surfaces.install(None)
    harness_surfaces.set_state_loader(None)
    yield
    harness_surfaces.install(None)
    harness_surfaces.set_state_loader(None)


def _install(tmp_path, body=LEDGER):
    path = tmp_path / "harness.py"
    path.write_text(body)
    harness_surfaces.install(path)
    assert harness_surfaces.status()["error"] is None


def _attempt(store, task_id, generation, score, digest=""):
    return store.record_attempt(
        task_id=task_id,
        agent_id=f"agent-{task_id}",
        generation=generation,
        model_output="done",
        native_score=score,
        trace_digest=digest,
    )


def _bundle(text):
    return {
        "transferable_insights": [{"text": text}],
        "confirmed_constraints": [],
        "rejected_hypotheses": [],
        "pitfalls": [],
        "checks": [],
        "next_steps": [],
    }


def _store(tmp_path):
    """Two generations: t1 fails then is solved, t2 fails twice, h1 is a hold-out."""
    store = KnowledgeStore(str(tmp_path / "k.sqlite"))
    for task_id in ("t1", "t2", "h1"):
        _attempt(store, task_id, 1, 0.0, digest="tried rule A" if task_id == "t1" else "")
    store.record_distillation(task_id="t1", generation=1, bundle=_bundle("use rule B"), scope="per_task")
    store.record_distillation(
        task_id=CROSS_TASK_SENTINEL, generation=1, bundle=_bundle("check every train pair"), scope="cross_task"
    )
    _attempt(store, "t1", 2, 1.0)
    _attempt(store, "t2", 2, 0.5)
    _attempt(store, "h1", 2, 1.0)
    return store


def _tables(path):
    conn = sqlite3.connect(path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    conn.close()
    return names


def test_default_keeps_no_state_and_leaves_the_store_alone(tmp_path):
    store = _store(tmp_path)
    assert harness_state.refresh(store, 1, ["t1", "t2"]) is None
    assert harness_state.rebuild(store, 2, ["t1", "t2"]) is None
    assert harness_surfaces.get_state() == {}
    assert store.load_harness_state() is None
    assert "harness_state" not in _tables(tmp_path / "k.sqlite")
    assert harness_state.update_state({"a": 1}, {"generation": 1, "tasks": {}}) == {"a": 1}


def test_feedback_has_outcomes_and_the_bundles_the_task_was_seeded_from(tmp_path):
    store = _store(tmp_path)
    first = harness_state.build_feedback(store, 1, ["t1", "t2"])
    assert first["generation"] == 1 and sorted(first["tasks"]) == ["t1", "t2"]
    assert first["tasks"]["t1"]["seeded"] == {"per_task": None, "cross_task": None}
    assert first["tasks"]["t1"]["attempts"][0]["trace_digest"] == "tried rule A"
    assert "trace_digest" not in first["tasks"]["t2"]["attempts"][0]

    second = harness_state.build_feedback(store, 2, ["t1", "t2"])
    t1, t2 = second["tasks"]["t1"], second["tasks"]["t2"]
    assert (t1["score"], t1["solved"], t2["score"], t2["solved"]) == (1.0, True, 0.5, False)
    assert t1["seeded"]["per_task"]["transferable_insights"][0]["text"] == "use rule B"
    assert t2["seeded"]["per_task"] is None
    # no target-conditioned cross-task bundle stored: both fall back to the shared one
    for task in (t1, t2):
        assert task["seeded"]["cross_task"]["transferable_insights"][0]["text"] == "check every train pair"
    assert "h1" not in second["tasks"]  # hold-outs are not in the caller's task list


def test_state_accumulates_across_generations_and_is_stored(tmp_path):
    store = _store(tmp_path)
    _install(tmp_path)
    assert harness_state.refresh(store, 1, ["t1", "t2"]) == {"generations": [1], "ledger": {}}
    state = harness_state.refresh(store, 2, ["t1", "t2"])
    assert state["generations"] == [1, 2]
    assert state["ledger"] == {
        "use rule B": {"shown": 1, "solved": 1},
        "check every train pair": {"shown": 2, "solved": 1},
    }
    assert harness_surfaces.get_state() == state
    assert store.load_harness_state() == (2, state)
    assert store.load_harness_state(max_generation=1) == (1, {"generations": [1], "ledger": {}})
    # a copy: mutating what a surface got does not change the stored state
    harness_surfaces.get_state()["ledger"].clear()
    assert harness_surfaces.get_state() == state


def test_distill_updates_the_state_before_distilling(tmp_path):
    store = _store(tmp_path)
    _install(tmp_path)
    seen = []

    def llm(*args, **kwargs):
        seen.append(harness_surfaces.get_state()["generations"])
        raise RuntimeError("no model in this test")

    inp = DistillInput(generation=1, task_ids=["t1", "t2"], knowledge_store=store, llm=llm)
    distiller.distill(inp, unsolved_task_ids=["t1"])
    assert store.load_harness_state() == (1, {"generations": [1], "ledger": {}})
    assert seen and all(generations == [1] for generations in seen)


def test_a_fresh_process_loads_the_stored_state(tmp_path):
    store = _store(tmp_path)
    _install(tmp_path)
    state = harness_state.rebuild(store, 2, ["t1", "t2"])
    harness_surfaces.set_state(None)  # what a resumed run starts with
    harness_state.bind(store)
    assert harness_surfaces.get_state() == state
    # without the surface the stored state of another harness is not handed out
    harness_surfaces.install(None)
    harness_state.bind(store)
    assert harness_surfaces.get_state() == {}


def test_binding_the_same_store_again_keeps_the_state_in_memory(tmp_path):
    store = _store(tmp_path)
    _install(tmp_path)
    harness_state.bind(store)
    harness_surfaces.set_state({"only": "in memory"})  # as left by refresh() in a live campaign
    harness_state.bind(store)  # the enrichment phase binds at every generation
    assert harness_surfaces.get_state() == {"only": "in memory"}
    other = KnowledgeStore(str(tmp_path / "other.sqlite"))
    harness_state.bind(other)  # a different store: its (absent) stored state applies
    assert harness_surfaces.get_state() == {}


def test_rebuild_replaces_states_left_by_another_harness(tmp_path):
    store = _store(tmp_path)
    store.record_harness_state(generation=1, state={"from": "another harness"})
    store.record_harness_state(generation=5, state={"from": "another harness"})
    _install(tmp_path)
    state = harness_state.rebuild(store, 2, ["t1", "t2"])
    assert state["generations"] == [1, 2]
    assert store.load_harness_state() == (2, state)


@pytest.mark.parametrize(
    "body",
    [
        "SURFACES = {'state.update': lambda default, state, feedback: ['not', 'a', 'dict']}",
        "SURFACES = {'state.update': lambda default, state, feedback: {'x': object()}}",
        "SURFACES = {'state.update': lambda default, state, feedback: {'x': 'y' * 300000}}",
        "def boom(default, state, feedback):\n    raise ValueError('bug')\nSURFACES = {'state.update': boom}",
    ],
)
def test_bad_results_keep_the_previous_state(tmp_path, body):
    store = _store(tmp_path)
    store.record_harness_state(generation=1, state={"kept": True})
    _install(tmp_path, body)
    assert harness_state.refresh(store, 2, ["t1", "t2"]) == {"kept": True}
    assert store.load_harness_state() == (2, {"kept": True})
