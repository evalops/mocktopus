# 🐙 Mocktopus

> Multi-armed mocks for LLM apps

**Mocktopus** is a drop-in replacement for OpenAI/Anthropic APIs, designed to make your LLM application tests fast, deterministic, and cost-free.

[![CI](https://github.com/evalops/mocktopus/actions/workflows/ci.yml/badge.svg)](https://github.com/evalops/mocktopus/actions)
[![PyPI](https://img.shields.io/pypi/v/mocktopus)](https://pypi.org/project/mocktopus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why Mocktopus?

Testing LLM applications is challenging:
- **Non-deterministic**: Same prompt, different responses
- **Expensive**: Every test run costs API credits
- **Slow**: API calls add latency to test suites
- **Network-dependent**: Can't run tests offline
- **Complex workflows**: Tool calls and streaming complicate testing

Mocktopus solves these problems by providing a local mock server that perfectly mimics LLM APIs.

## Features

✅ **Drop-in replacement** - Just change your base URL
✅ **Deterministic responses** - Same input → same output
✅ **Tool/function calling** - Full support for complex workflows
✅ **Streaming** - Server-sent events (SSE) support
✅ **Multiple providers** - OpenAI and Anthropic compatible
✅ **Zero cost** - No API charges for tests
✅ **Fast** - No network latency
✅ **Offline** - Run tests without internet

## Installation

```bash
pip install mocktopus
```

## Quick Start

### 1. Create a scenario file (`scenario.yaml`):

```yaml
version: 1
rules:
  - type: llm.openai
    when:
      model: "gpt-4*"
      messages_contains: "hello"
    respond:
      content: "Hello! How can I help you today?"
```

### 2. Start the mock server:

```bash
mocktopus serve -s scenario.yaml
```

### 3. Point your app to Mocktopus:

```python
from openai import OpenAI

# Instead of the real API:
# client = OpenAI(api_key="sk-...")

# Use Mocktopus:
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="mock-key"  # Any string works
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "hello"}]
)
print(response.choices[0].message.content)
# Output: "Hello! How can I help you today?"
```

## Usage Modes

### Mock Mode (Default)
Use predefined YAML scenarios for deterministic responses:

```bash
mocktopus serve -s examples/chat-basic.yaml
```

### Record Mode (Coming Soon)
Proxy and record real API calls for later replay:

```bash
mocktopus serve --mode record --recordings-dir ./recordings
```

### Replay Mode (Coming Soon)
Replay previously recorded API interactions:

```bash
mocktopus serve --mode replay --recordings-dir ./recordings
```

## Scenario Examples

### Basic Chat Response

```yaml
version: 1
rules:
  - type: llm.openai
    when:
      messages_contains: "weather"
    respond:
      content: "It's sunny today!"
```

### Function Calling

```yaml
version: 1
rules:
  - type: llm.openai
    when:
      messages_contains: "weather"
    respond:
      tool_calls:
        - id: "call_123"
          type: "function"
          function:
            name: "get_weather"
            arguments: '{"location": "San Francisco"}'
```

### Streaming Response

```yaml
version: 1
rules:
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "This will be streamed..."
      delay_ms: 50  # Delay between chunks
      chunk_size: 5  # Characters per chunk
```

### Limited Usage

```yaml
version: 1
rules:
  - type: llm.openai
    when:
      messages_contains: "test"
    times: 3  # Only responds 3 times
    respond:
      content: "Limited response"
```

## CLI Commands

### Start Server
```bash
# Basic usage
mocktopus serve -s scenario.yaml

# Custom port
mocktopus serve -s scenario.yaml -p 9000

# Verbose logging
mocktopus serve -s scenario.yaml -v
```

### Test Scenarios
```bash
# Validate a scenario file
mocktopus validate scenario.yaml

# Simulate a request without starting server
mocktopus simulate -s scenario.yaml --prompt "Hello"

# Generate example scenarios
mocktopus example --type basic > my-scenario.yaml
mocktopus example --type tools > tools-scenario.yaml
```

## Testing with Mocktopus

### Pytest Integration

```python
import pytest
from mocktopus import use_mocktopus

def test_my_llm_app(use_mocktopus):
    # Load scenario
    use_mocktopus.load_yaml("tests/scenarios/test.yaml")

    # Get a client
    client = use_mocktopus.openai_client()

    # Test your app
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}]
    )
    assert "expected" in response.choices[0].message.content
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e .
      - run: mocktopus serve -s tests/scenarios.yaml &
      - run: pytest  # Your tests hit localhost:8080
```

## Advanced Features

### Pattern Matching

Mocktopus supports multiple matching strategies:

- **Exact match**: `messages_contains: "exact phrase"`
- **Regex**: `messages_regex: "\\d+ items?"`
- **Glob**: `model: "gpt-4*"`

### Response Configuration

```yaml
respond:
  content: "Response text"
  delay_ms: 100  # Simulate latency
  usage:
    input_tokens: 10
    output_tokens: 20
  # For streaming
  chunk_size: 10  # Characters per chunk
```

## Roadmap

- [x] OpenAI chat completions API
- [x] Streaming support (SSE)
- [x] Function/tool calling
- [x] Anthropic messages API
- [ ] Recording & replay
- [ ] Embeddings API
- [ ] Assistants API
- [ ] Image generation
- [ ] Semantic similarity matching
- [ ] Response templating
- [ ] Load testing mode

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT - See [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/evalops/mocktopus)
- [PyPI Package](https://pypi.org/project/mocktopus/)
- [Documentation](https://github.com/evalops/mocktopus/wiki)
- [Issue Tracker](https://github.com/evalops/mocktopus/issues)

---

Made with 🐙 by [EvalOps](https://github.com/evalops)