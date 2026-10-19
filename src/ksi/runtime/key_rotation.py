"""Switchable provider API key (opt-in).

When ``KSI_API_KEY_STATE`` names a file whose content is the name of a
variable in the ``KSI_API_KEY_ENV_FILE`` dotenv file (e.g. ``key_``), the
active key is read from there at every solver-container launch and before
every host-side LLM call. An external watchdog
(``scripts/openlux_keyguard.py`` in the KHSI repo) rewrites the state file
when a key runs out of quota, so a run moves to the backup key without
restarting. The key value itself only ever lives in the dotenv file.

Unset (the default), nothing changes: keys come from the provider profile.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

STATE_VAR = "KSI_API_KEY_STATE"
ENV_FILE_VAR = "KSI_API_KEY_ENV_FILE"

# Provider replies meaning "this key/account has no quota left" (OpenAI,
# new-api/one-api gateways, Anthropic). Rate limits without these words are not
# quota exhaustion.
QUOTA_ERROR = re.compile(
    r"insufficient[_ ](user_)?quota|quota[_ ](exceeded|is not enough|not enough)|exceeded your current quota"
    r"|余额不足|额度不足|额度已用尽|credit balance is too low|insufficient[_ ]balance|payment required",
    re.IGNORECASE,
)

_lock = threading.Lock()
_cache: dict[str, object] = {"stamp": None, "key": None}


def enabled() -> bool:
    return bool(os.environ.get(STATE_VAR) and os.environ.get(ENV_FILE_VAR))


def is_quota_error(text: str) -> bool:
    return bool(QUOTA_ERROR.search(text or ""))


def active_key() -> str | None:
    """The key the state file currently selects, or None when rotation is off or unreadable."""
    if not enabled():
        return None
    state, env_file = Path(os.environ[STATE_VAR]), Path(os.environ[ENV_FILE_VAR])
    try:
        stamp = (state.stat().st_mtime_ns, env_file.stat().st_mtime_ns)
    except OSError:
        return None
    with _lock:
        if _cache["stamp"] == stamp:
            return _cache["key"]  # type: ignore[return-value]
    try:
        name = state.read_text().strip()
        key = None
        for line in env_file.read_text().splitlines():
            if name and line.startswith(f"{name}="):
                key = line.split("=", 1)[1].strip().strip("'\"") or None
                break
    except OSError:
        return None
    with _lock:
        _cache.update(stamp=stamp, key=key)
    return key


def apply_to_env(env: dict[str, str]) -> dict[str, str]:
    """Overwrite whichever provider key ``env`` carries with the active key."""
    key = active_key()
    if key:
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if env.get(var):
                env[var] = key
    return env
