from __future__ import annotations

import itertools
import time
from contextlib import ContextDecorator
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from .core import Scenario, Rule


# ----------------------------- Simple response objects -----------------------------

class _Usage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: Optional[int] = None):
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens


class _Message:
    def __init__(self, role: str, content: Optional[str], tool_calls: Optional[List[Dict[str, Any]]] = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, message: _Message, finish_reason: str = "stop", index: int = 0):
        self.message = message
        self.finish_reason = finish_reason
        self.index = index
        # for streaming parity:
        self.delta = SimpleNamespace(content=None)


class _ChatCompletion:
    def __init__(self, *, id: str, model: str, choices: List[_Choice], usage: _Usage):
        self.id = id
        self.model = model
        self.choices = choices
        self.usage = usage

    # lightweight duck-typing helpers seen in some SDKs
    def dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "finish_reason": c.finish_reason,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        "tool_calls": c.message.tool_calls,
                    },
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }

    # OpenAI Pydantic compatibility helpers:
    def model_dump(self) -> Dict[str, Any]:
        return self.dict()

    def model_dump_json(self) -> str:
        import json
        return json.dumps(self.dict())


def _stream_from_text(text: str, *, delay_ms: int = 0, chunk_size: int = 12) -> Iterable[Any]:
    """Yield a sequence of events that looks like OpenAI streaming responses."""
    for i, start in enumerate(range(0, len(text), chunk_size)):
        chunk = text[start : start + chunk_size]
        event = SimpleNamespace()
        delta = SimpleNamespace(content=chunk)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        event.choices = [choice]
        yield event
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
    # final with finish_reason
    done = SimpleNamespace()
    choice_done = SimpleNamespace(index=0, delta=SimpleNamespace(content=None), finish_reason="stop")
    done.choices = [choice_done]
    yield done


# ----------------------------- OpenAI Stub Client ----------------------------------

class OpenAIStubClient:
    """Small, dependency-injected stub that mimics enough of the OpenAI client for tests."""

    class _Completions:
        def __init__(self, scenario: Scenario):
            self._scenario = scenario

        def create(self, *, model: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs: Any):
            rule, respond = self._scenario.find_llm(model=model, messages=messages)
            if not rule:
                raise KeyError("No Mocktopus rule matched for the given model/messages.")
            rule.consume()

            content: str = respond.get("content", "")
            tool_calls = respond.get("tool_calls") or []
            usage = respond.get("usage") or {}
            if stream:
                return _stream_from_text(content, delay_ms=respond.get("delay_ms", 0), chunk_size=respond.get("chunk_size", 12))

            choice = _Choice(_Message("assistant", content, tool_calls=tool_calls), "stop", 0)
            return _ChatCompletion(
                id=respond.get("id", "mocktopus-chatcmpl-1"),
                model=model,
                choices=[choice],
                usage=_Usage(
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    total_tokens=usage.get("total_tokens"),
                ),
            )

    class _Chat:
        def __init__(self, scenario: Scenario):
            self.completions = OpenAIStubClient._Completions(scenario)

    def __init__(self, scenario: Scenario):
        self.chat = OpenAIStubClient._Chat(scenario)


# ----------------------------- OpenAI SDK Patcher ----------------------------------

class patch_openai(ContextDecorator):
    """
    Best-effort monkey-patcher for the OpenAI Python SDK to route chat.completions.create()
    through the Mocktopus Scenario. If the SDK layout isn't supported, it becomes a no-op.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self._patched = False
        self._original = None
        self._target_class = None

    def __enter__(self):
        try:
            import importlib
            mod = importlib.import_module("openai.resources.chat.completions")
            self._target_class = getattr(mod, "Completions", None)
            if not self._target_class:
                return self
            original = getattr(self._target_class, "create")
            self._original = original

            scenario = self.scenario

            def fake_create(this, *args, **kwargs):
                model = kwargs.get("model") or ""
                messages = kwargs.get("messages") or []
                stream = kwargs.get("stream") or False
                rule, respond = scenario.find_llm(model=model, messages=messages)
                if not rule:
                    raise KeyError("No Mocktopus rule matched for the given model/messages.")
                rule.consume()
                content: str = respond.get("content", "")
                if stream:
                    delay_ms = respond.get("delay_ms", 0)
                    chunk_size = respond.get("chunk_size", 12)
                    return _stream_from_text(content, delay_ms=delay_ms, chunk_size=chunk_size)
                # synthesize a minimal response object
                tool_calls = respond.get("tool_calls") or []
                usage = respond.get("usage") or {}
                choice = _Choice(_Message("assistant", content, tool_calls=tool_calls), "stop", 0)
                return _ChatCompletion(
                    id=respond.get("id", "mocktopus-chatcmpl-1"),
                    model=model,
                    choices=[choice],
                    usage=_Usage(
                        input_tokens=int(usage.get("input_tokens", 0)),
                        output_tokens=int(usage.get("output_tokens", 0)),
                        total_tokens=usage.get("total_tokens"),
                    ),
                )

            setattr(self._target_class, "create", fake_create)
            self._patched = True
        except Exception:
            # leave as no-op if anything fails
            self._patched = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._patched and self._target_class and self._original:
            try:
                setattr(self._target_class, "create", self._original)
            except Exception:
                pass
        return False
