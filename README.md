# Mocktopus

Mocktopus is a local, deterministic mock server for LLM application tests. It speaks the
OpenAI-style chat completions and embeddings endpoints well enough for CI, fixture-based
integration tests, and local development without live model calls.

[![CI](https://github.com/evalops/mocktopus/actions/workflows/ci.yml/badge.svg)](https://github.com/evalops/mocktopus/actions/workflows/ci.yml)
[![Bazel RBE](https://github.com/evalops/mocktopus/actions/workflows/bazel-rbe.yml/badge.svg)](https://github.com/evalops/mocktopus/actions/workflows/bazel-rbe.yml)
[![PyPI](https://img.shields.io/pypi/v/mocktopus)](https://pypi.org/project/mocktopus/)

## What It Does

- Serves deterministic chat completion responses from YAML scenarios.
- Supports OpenAI-style streaming, tool calls, embeddings, and common error responses.
- Tracks estimated cost savings for mocked requests.
- Provides a small Python stub client for fast unit tests that do not need an HTTP server.
- Runs in local pytest, Bazel, and EvalOps Bazel RBE workflows.

Record and replay modes exist as server modes, but the most reliable path today is
scenario-driven mock mode.

## Install

```bash
pip install mocktopus
```

For development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

Create `scenario.yaml`:

```yaml
version: 1
rules:
  - type: llm.openai
    when:
      model: "gpt-4*"
      messages_contains: "hello"
    respond:
      content: "Hello from Mocktopus."
      usage:
        input_tokens: 3
        output_tokens: 4
```

Start the server:

```bash
mocktopus serve -s scenario.yaml
```

Point your app at the local OpenAI-compatible base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="mock-key")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
```

## Scenario Rules

Rules match in file order. A rule can return either `respond` or `error`.

```yaml
version: 1
rules:
  - type: llm.openai
    when:
      messages_contains: "weather"
    respond:
      content: "Sunny, 72F."

  - type: llm.openai
    when:
      messages_contains: "rate limit"
    error:
      error_type: rate_limit
      message: "Rate limit exceeded"
      status_code: 429
      retry_after: 60
```

Supported match fields:

- `model`: glob pattern such as `gpt-4*`
- `messages_contains`: substring match against the user message
- `messages_regex`: regular expression over message text
- `endpoint`: endpoint selector such as `/v1/embeddings`
- `times`: maximum uses for the rule before the next matching rule is tried

Supported error types are `rate_limit`, `authentication`, `invalid_request`, `timeout`,
`content_filter`, and `server_error`.

## Embeddings

Embeddings can be pinned directly in a scenario:

```yaml
version: 1
rules:
  - type: embeddings
    when:
      endpoint: "/v1/embeddings"
    respond:
      embeddings:
        - embedding: [0.1, 0.2, -0.3]
      usage:
        input_tokens: 7
```

If no embedding vector is provided, Mocktopus generates a deterministic vector from the
input text and model name.

## CLI

```bash
mocktopus serve -s examples/chat-basic.yaml
mocktopus serve -s scenario.yaml --port 9000
mocktopus validate scenario.yaml
mocktopus explain -s scenario.yaml --prompt "hello"
mocktopus simulate -s scenario.yaml --prompt "hello"
```

## Tests And Builds

Local checks:

```bash
PYTHONPATH=src python3 -m pytest -q
make bazel-check
```

Remote execution smoke:

```bash
make bazel-rbe-smoke
```

The `Bazel RBE` GitHub Actions workflow runs on the EvalOps `bazel-rbe-dev` farm when
`BAZEL_RBE_ENABLED=true` is set for the repository. It uses the `evalops-mocktopus-rbe`
and `bazel-rbe` self-hosted labels.

## Repository Layout

```text
src/mocktopus/
  cli.py              command-line interface
  core.py             scenario schema, YAML loading, and rule matching
  server.py           aiohttp mock API server
  stub_openai.py      in-process OpenAI-like test client
tests/                unit and integration coverage
examples/             scenario examples
```
