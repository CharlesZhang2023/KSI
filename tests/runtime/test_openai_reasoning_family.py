"""Reasoning-family detection for host-side OpenAI Responses calls."""

import pytest

from ksi.runtime.llm import _is_openai_reasoning_model


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5", "gpt-6-luna", "o3-mini", "o4-mini", "o1"])
def test_reasoning_family_models(model):
    assert _is_openai_reasoning_model(model)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1", "", "qwen3-32b"])
def test_non_reasoning_models(model):
    assert not _is_openai_reasoning_model(model)
