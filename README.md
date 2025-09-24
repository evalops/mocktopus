# Mocktopus 🐙

**Multi‑armed mocks for LLM apps.** Deterministic, dataset‑driven mocks for OpenAI‑style chat
completions (plus tool calls & streaming simulation). Designed for evals, CI, and reproducible tests.

## Why
- Make flaky LLM tests deterministic.
- Run evals offline and in CI without credentials.
- Record once, replay many times (golden fixtures).
- Simulate streaming and tool calls without hitting real APIs.

## Features (MVP)
- 🧪 `Scenario` that loads rules from YAML and returns canned LLM responses.
- 🧵 Streaming simulation compatible with `stream=True` style iteration.
- 🧰 Tool call stubs (`assistant.tool_calls`) with optional structured args.
- 🧩 Two ways to use:
  1. **Dependency‑injected client**: `OpenAIStubClient(scenario)`
  2. **Monkey‑patch** (best‑effort): `with patch_openai(scenario): ...`

> Patching SDK internals can be brittle across versions. Prefer dependency injection for reliability.

Roadmap: HTTP fixtures (VCR‑like), vector/RAG stubs, Anthropic adapter, record mode.

## Install
```bash
pip install -e ".[dev]"  # when cloned locally
# or just install from source once published:
# pip install mocktopus
```

## Quick start

1) Define a fixture:

```yaml
# examples/haiku.yaml
version: 1
rules:
  - type: llm.openai
    when:
      messages_contains: "haiku"
      model: "*"
    respond:
      content: "Silent bay at dusk\nEight arms fold into the deep\nTides keep time for stars."
      usage:
        input_tokens: 12
        output_tokens: 17
      stream: false
```

2) Use the dependency‑injected stub client:

```python
from mocktopus import Scenario, OpenAIStubClient, load_yaml

scenario = load_yaml("examples/haiku.yaml")
client = OpenAIStubClient(scenario)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku about an octopus"}],
)

print(resp.choices[0].message.content)
# -> Silent bay at dusk ...
```

3) Or (beta) patch the OpenAI SDK dynamically:

```python
from mocktopus import load_yaml, patch_openai
from openai import OpenAI

scenario = load_yaml("examples/haiku.yaml")
with patch_openai(scenario):
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a haiku about an octopus"}],
    )
    print(resp.choices[0].message.content)
```

4) Streaming simulation:

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "haiku please"}],
    stream=True,
)
for event in resp:
    delta = event.choices[0].delta.content or ""
    print(delta, end="")
```

## Pytest usage

Add to your test conftest:
```python
pytest_plugins = ["mocktopus.pytest_plugin"]
```

Example test:
```python
def test_haiku(use_mocktopus):
    use_mocktopus.load_yaml("examples/haiku.yaml")
    client = use_mocktopus.openai_client()  # OpenAIStubClient bound to the scenario
    out = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "haiku"}],
    )
    assert "Silent bay" in out.choices[0].message.content
```

## CLI

```bash
mocktopus simulate --fixture examples/haiku.yaml --prompt "haiku about octopus"
```

## Caveats
- The OpenAI SDK patcher targets modern `openai` Python SDKs and may break across versions. Prefer the stub client where possible.
- YAML matching is currently simple (substring + model glob). Extend as needed.

## License
MIT
