"""The knowledge harness's own state across generations (surface ``state.update``).

The default pipeline recomputes everything from the windowed raw history each
generation and keeps no record of what its knowledge led to. A harness that
overrides ``state.update`` gets, once per generation and before distillation,
the state it returned one generation earlier plus feedback on the generation
that just ended: per task, the outcome of its attempts and the bundles that
task had been seeded from. Whatever dict it returns is stored in the knowledge
database and is readable from every other surface through
``ksi.harness_surfaces.get_state()`` — enough for a ledger of which items were
shown and what followed, or for carrying knowledge forward instead of
re-deriving it.

With no harness, or a harness that does not override the surface, nothing here
queries or writes the store.
"""

from __future__ import annotations

import json
import logging
import weakref
from typing import Any

from . import harness_surfaces
from .harness_surfaces import surface

log = logging.getLogger(__name__)

SURFACE = "state.update"
STATE_MAX_BYTES = 256 * 1024
_SOLVED_SCORE = 1.0


@surface(SURFACE)
def update_state(state: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    """The harness's state after seeing ``feedback`` (default: unchanged)."""
    return state


_bound: tuple[Any, str | None, Any] | None = None  # (weakref to the store, experiment, loader)


def bind(knowledge_store: Any, *, experiment: str | None = None) -> None:
    """Let ``harness_surfaces.get_state()`` fetch the stored state from ``knowledge_store``.

    Called wherever a process starts using a store for seeding or replay (the
    enrichment phase, the offline replay), so a process that resumes a run and
    only seeds and attempts still sees the state an earlier process computed.
    Binding the same store again is a no-op, so the state held in memory by a
    running campaign is not dropped every generation.
    """
    global _bound
    if (
        _bound is not None
        and _bound[0]() is knowledge_store
        and _bound[1] == experiment
        and harness_surfaces._harness_state_loader is _bound[2]
    ):
        return
    ref = weakref.ref(knowledge_store)

    def _load() -> dict[str, Any] | None:
        store = ref()
        if store is None or not harness_surfaces.overridden(SURFACE):
            return None
        loaded = store.load_harness_state(experiment=experiment)
        return loaded[1] if loaded else None

    _bound = (ref, experiment, _load)
    harness_surfaces.set_state_loader(_load)


def build_feedback(
    knowledge_store: Any,
    generation: int,
    task_ids: list[str],
    *,
    experiment: str | None = None,
) -> dict[str, Any]:
    """What happened at ``generation`` for the tasks in ``task_ids`` (hold-outs excluded by the caller)."""
    from .memory.knowledge_store import CROSS_TASK_SENTINEL

    wanted = {str(t) for t in task_ids}
    tasks: dict[str, dict[str, Any]] = {}
    rows = knowledge_store.query_generation(int(generation), entry_types=["attempt"], experiment=experiment)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id not in wanted:
            continue
        content = row.get("content") if isinstance(row.get("content"), dict) else {}
        score = float(row.get("native_score") or 0.0)
        attempt: dict[str, Any] = {"id": row.get("id"), "agent_id": row.get("agent_id"), "score": score}
        if content.get("trace_digest"):
            attempt["trace_digest"] = content["trace_digest"]
        entry = tasks.setdefault(task_id, {"score": 0.0, "solved": False, "attempts": []})
        entry["attempts"].append(attempt)
        entry["score"] = max(entry["score"], score)
        entry["solved"] = entry["solved"] or score >= _SOLVED_SCORE

    if tasks and int(generation) > 1:
        previous = int(generation) - 1
        ids = sorted(tasks)
        per_task = knowledge_store.load_distillations_batch(
            generation=previous, task_ids=ids, scope="per_task", experiment=experiment
        )
        cross_task = knowledge_store.load_distillations_batch(
            generation=previous, task_ids=ids, scope="cross_task", experiment=experiment
        )
        shared = knowledge_store.load_distillation(
            generation=previous, task_id=CROSS_TASK_SENTINEL, scope="cross_task", experiment=experiment
        )
    else:
        per_task, cross_task, shared = {}, {}, None
    for task_id, entry in tasks.items():
        entry["seeded"] = {
            "per_task": per_task.get(task_id),
            "cross_task": cross_task.get(task_id) or shared,
        }
    return {"generation": int(generation), "tasks": tasks}


def refresh(
    knowledge_store: Any,
    generation: int,
    task_ids: list[str],
    *,
    experiment: str | None = None,
) -> dict[str, Any] | None:
    """Run ``state.update`` for ``generation`` and store the result.

    Returns the new state, or None when the active harness keeps no state. A
    result that is not a JSON object within ``STATE_MAX_BYTES`` is rejected:
    the previous state stays in force and the rejection is logged.
    """
    if knowledge_store is None or not harness_surfaces.overridden(SURFACE):
        return None
    loaded = knowledge_store.load_harness_state(max_generation=int(generation) - 1, experiment=experiment)
    previous: dict[str, Any] = loaded[1] if loaded else {}
    feedback = build_feedback(knowledge_store, generation, task_ids, experiment=experiment)
    result = update_state(json.loads(json.dumps(previous)), feedback)
    state = previous
    problem = _invalid(result)
    if problem:
        log.error(
            "[HARNESS] state.update result for gen=%d rejected (%s); keeping the previous state", generation, problem
        )
    else:
        state = json.loads(json.dumps(result))
    knowledge_store.record_harness_state(
        generation=int(generation),
        state=state,
        harness_digest=harness_surfaces.status().get("digest"),
        experiment=experiment,
    )
    harness_surfaces.set_state(state)
    log.info(
        "[HARNESS] state.update gen=%d: %d task(s) of feedback, state %d bytes",
        generation,
        len(feedback["tasks"]),
        len(json.dumps(state)),
    )
    return state


def rebuild(
    knowledge_store: Any,
    upto_generation: int,
    task_ids: list[str],
    *,
    experiment: str | None = None,
) -> dict[str, Any] | None:
    """Recompute the state for generations 1..``upto_generation`` from a finished run.

    For offline replay: the stored states (if any) came from whatever harness
    produced the run, so they are dropped and rebuilt with the active one. The
    feedback is then about the knowledge the *recorded* run injected, not what
    the active harness would have injected.
    """
    if knowledge_store is None:
        return None
    knowledge_store.clear_harness_state(experiment=experiment)
    harness_surfaces.set_state(None)
    state: dict[str, Any] | None = None
    for generation in range(1, int(upto_generation) + 1):
        state = refresh(knowledge_store, generation, task_ids, experiment=experiment)
    return state


def _invalid(result: Any) -> str:
    if not isinstance(result, dict):
        return f"returned {type(result).__name__}, not a dict"
    try:
        size = len(json.dumps(result))
    except (TypeError, ValueError) as exc:
        return f"not JSON-serialisable: {exc}"
    if size > STATE_MAX_BYTES:
        return f"{size} bytes as JSON, limit {STATE_MAX_BYTES}"
    return ""
