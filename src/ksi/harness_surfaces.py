"""Replaceable knowledge-harness surfaces (KHSI Stage 1).

A *surface* is one function in the knowledge-processing pipeline that a
knowledge harness may replace: how forums are prompted, how bundles are
distilled and filtered, how cross-task posts are ranked for a target, and how
knowledge is rendered into an agent's MEMORY.md. Each is marked with
``@surface(name)`` at its definition, so every call site goes through it.

With no harness installed every surface calls its original function directly;
the default pipeline is unchanged. A harness is a Python file that defines::

    SURFACES = {
        "distill.per_task.prompt": my_prompt,   # called as my_prompt(default, *args, **kwargs)
        ...
    }

An override receives the original implementation as its first argument so it
can wrap, post-process or replace it. The harness file is named by
``KSI_KNOWLEDGE_HARNESS`` (or installed with :func:`install`) and is reloaded
whenever its content changes, so a harness edited between generations takes
effect at the next surface call. A harness that fails to load is reported and
the defaults are used; the failure is recorded in :func:`status`.

This module must stay dependency-free: it is imported by low-level modules.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

ENV_VAR = "KSI_KNOWLEDGE_HARNESS"

F = TypeVar("F", bound=Callable[..., Any])

# Every replaceable surface, with what it controls. Kept here (not only in the
# decorators) so a harness can be validated before all modules are imported.
KNOWN_SURFACES: dict[str, str] = {
    "forum.per_task.prompt": "ksi.forum.prompt.build_per_task_discussion_parts: per-task forum prompt",
    "forum.cross_task.prompt": "ksi.forum.prompt.build_cross_task_discussion_parts: cross-task forum prompt",
    "distill.per_task.prompt": "ksi.distillation.prompts.build_per_task_distill_prompt: per-task distillation prompt",
    "distill.cross_task.prompt": (
        "ksi.distillation.prompts.build_cross_task_distill_prompt_parts: cross-task distillation prompt"
    ),
    "distill.bundle_filter": "ksi.distillation.per_task.dedupe_bundle_items: which distilled items are kept",
    "retrieve.cross_task.relevance": (
        "ksi.memory.cross_task_context.target_relevance: ranking of cross-task posts for a target task"
    ),
    "seed.memory_md": "ksi.runtime.seeding.seed_package_to_memory_md: the MEMORY.md an agent receives",
    "seed.render_item": "ksi.runtime.seeding._render_bundle_item: how one knowledge item is rendered",
}

DEFAULTS: dict[str, Callable[..., Any]] = {}

_lock = threading.Lock()
_state: dict[str, Any] = {
    "path": None,  # explicit install() path; falls back to the env var
    "digest": None,
    "overrides": {},
    "error": None,
    "call_errors": {},  # surface -> count of override exceptions (default used instead)
}


def surface(name: str) -> Callable[[F], F]:
    """Mark ``default`` as the replaceable implementation of surface ``name``."""

    def decorate(default: F) -> F:
        if name not in KNOWN_SURFACES:
            raise ValueError(f"surface {name!r} is not listed in KNOWN_SURFACES")
        if name in DEFAULTS:
            raise ValueError(f"surface {name!r} defined twice")
        DEFAULTS[name] = default

        @functools.wraps(default)
        def dispatch(*args: Any, **kwargs: Any) -> Any:
            override = _overrides().get(name)
            if override is None:
                return default(*args, **kwargs)
            from_default: list[BaseException] = []

            def tracked_default(*a: Any, **kw: Any) -> Any:
                try:
                    return default(*a, **kw)
                except BaseException as exc:
                    from_default.append(exc)
                    raise

            try:
                return override(tracked_default, *args, **kwargs)
            except Exception as exc:
                if from_default and exc is from_default[-1]:
                    raise  # the original implementation failed; not the harness's fault
                _record_call_error(name, exc)
                return default(*args, **kwargs)

        dispatch.__ksi_surface__ = name  # type: ignore[attr-defined]
        return dispatch  # type: ignore[return-value]

    return decorate


def _record_call_error(name: str, exc: Exception) -> None:
    with _lock:
        _state["call_errors"][name] = _state["call_errors"].get(name, 0) + 1
        count = _state["call_errors"][name]
    if count <= 3:
        log.error("[HARNESS] override of %s raised %s: %s; using the default", name, type(exc).__name__, exc)


def install(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Use the harness at ``path`` (None restores the env var / defaults)."""
    with _lock:
        _state.update(path=None if path is None else str(path), digest=None, overrides={}, error=None, call_errors={})
    _overrides()
    return status()


def status() -> dict[str, Any]:
    """Which harness is active: path, content digest, overridden surfaces, last load error."""
    _overrides()
    with _lock:
        return {
            "path": _active_path(),
            "digest": _state["digest"],
            "overrides": sorted(_state["overrides"]),
            "error": _state["error"],
            "call_errors": dict(_state["call_errors"]),
        }


def _active_path() -> str | None:
    return _state["path"] or os.environ.get(ENV_VAR) or None


def _overrides() -> dict[str, Callable[..., Any]]:
    path = _active_path()
    if not path:
        if _state["overrides"] or _state["digest"]:
            with _lock:
                _state.update(digest=None, overrides={}, error=None)
        return _state["overrides"]
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        _fail(f"cannot read knowledge harness {path}: {exc}")
        return _state["overrides"]
    digest = hashlib.sha256(data).hexdigest()
    if digest == _state["digest"]:
        return _state["overrides"]
    with _lock:
        if digest == _state["digest"]:
            return _state["overrides"]
        try:
            overrides = _load(path, digest)
        except Exception as exc:  # noqa: BLE001 - any harness bug falls back to defaults
            _state.update(digest=digest, overrides={}, error=f"{type(exc).__name__}: {exc}")
            log.error("[HARNESS] failed to load %s (%s); using default surfaces", path, exc)
            return _state["overrides"]
        _state.update(digest=digest, overrides=overrides, error=None)
        log.info("[HARNESS] loaded %s sha256=%s surfaces=%s", path, digest[:12], sorted(overrides))
        return overrides


def _fail(message: str) -> None:
    with _lock:
        if _state["error"] != message:
            log.error("[HARNESS] %s; using default surfaces", message)
        _state.update(digest=None, overrides={}, error=message)


def _load(path: str, digest: str) -> dict[str, Callable[..., Any]]:
    spec = importlib.util.spec_from_file_location(f"ksi_knowledge_harness_{digest[:12]}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"not a Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    surfaces = getattr(module, "SURFACES", None)
    if not isinstance(surfaces, dict):
        raise TypeError("harness must define SURFACES = {name: callable}")
    unknown = sorted(set(surfaces) - set(KNOWN_SURFACES))
    if unknown:
        raise KeyError(f"unknown surfaces {unknown}; known: {sorted(KNOWN_SURFACES)}")
    bad = sorted(name for name, fn in surfaces.items() if not callable(fn))
    if bad:
        raise TypeError(f"surfaces are not callable: {bad}")
    return dict(surfaces)
