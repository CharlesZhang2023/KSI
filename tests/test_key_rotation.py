"""Opt-in provider key rotation (ksi.runtime.key_rotation)."""

import os

import pytest

from ksi.orchestrator.task_retry import _is_retryable_task_error
from ksi.runtime import key_rotation
from ksi.runtime.container_host import _build_runner_env
from ksi.runtime.llm import OpenAILLMCaller


@pytest.fixture
def rotation(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("key=sk-first\nkey_=sk-second\nother=x\n")
    state = tmp_path / "active"
    state.write_text("key\n")
    monkeypatch.setenv(key_rotation.STATE_VAR, str(state))
    monkeypatch.setenv(key_rotation.ENV_FILE_VAR, str(env_file))
    return state


def _switch(state, name):
    state.write_text(name + "\n")
    st = state.stat()
    os.utime(state, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # mtime granularity


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(key_rotation.STATE_VAR, raising=False)
    assert not key_rotation.enabled() and key_rotation.active_key() is None
    env = {"OPENAI_API_KEY": "sk-profile"}
    assert key_rotation.apply_to_env(dict(env)) == env
    assert _build_runner_env(env, 600)["OPENAI_API_KEY"] == "sk-profile"
    assert not _is_retryable_task_error(RuntimeError("Error 403: insufficient_user_quota"))


def test_state_file_selects_key_and_follows_switches(rotation):
    assert key_rotation.active_key() == "sk-first"
    _switch(rotation, "key_")
    assert key_rotation.active_key() == "sk-second"
    env = key_rotation.apply_to_env({"OPENAI_API_KEY": "sk-profile", "MODEL": "m"})
    assert env == {"OPENAI_API_KEY": "sk-second", "MODEL": "m"}
    assert _build_runner_env({"OPENAI_API_KEY": "sk-profile"}, 600)["OPENAI_API_KEY"] == "sk-second"
    _switch(rotation, "missing")
    assert key_rotation.active_key() is None


def test_host_caller_rebuilds_client_on_switch(rotation, monkeypatch):
    for var in ("all_proxy", "ALL_PROXY", "https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(var, raising=False)
    caller = OpenAILLMCaller(model="m", api_key="sk-profile")
    caller._refresh_key()
    assert caller._client_key == "sk-first" and caller._client.api_key == "sk-first"
    _switch(rotation, "key_")
    caller._refresh_key()
    assert caller._client.api_key == "sk-second"


def test_quota_errors_retry_only_with_rotation(rotation):
    assert key_rotation.is_quota_error("用户额度不足, 剩余额度: 0")
    assert not key_rotation.is_quota_error("429 rate limit exceeded, retry later")
    assert _is_retryable_task_error(RuntimeError("Error code: 403 - insufficient_user_quota"))
