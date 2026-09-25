"""Opt-in thinking policy and thinking-block-safe text for AnthropicLLMCaller."""

from types import SimpleNamespace

from ksi.runtime.llm import AnthropicLLMCaller


class _FakeMessages:
    def __init__(self, content):
        self.content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        usage = SimpleNamespace(
            input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
        )
        return SimpleNamespace(content=self.content, usage=usage)


def _caller(content):
    caller = AnthropicLLMCaller.__new__(AnthropicLLMCaller)
    AnthropicLLMCaller.__init__(caller, model="deepseek-flash", api_key="k")
    fake = _FakeMessages(content)
    caller._client = SimpleNamespace(messages=fake)
    return caller, fake


def test_thinking_disabled_only_when_flag_set(monkeypatch):
    caller, fake = _caller([SimpleNamespace(type="text", text="ok")])
    monkeypatch.delenv("KSI_ANTHROPIC_DISABLE_THINKING", raising=False)
    caller.call(system="s", user="u")
    assert "thinking" not in fake.last_kwargs
    monkeypatch.setenv("KSI_ANTHROPIC_DISABLE_THINKING", "1")
    caller.call(system="s", user="u")
    assert fake.last_kwargs["thinking"] == {"type": "disabled"}


def test_text_skips_leading_thinking_block(monkeypatch):
    monkeypatch.delenv("KSI_ANTHROPIC_DISABLE_THINKING", raising=False)
    caller, _ = _caller([SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text="answer")])
    assert caller.call(system="s", user="u").text == "answer"
