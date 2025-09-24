from __future__ import annotations

import sys
import logging
from typing import List, Dict, Any
import click

from .core import Scenario, load_yaml
from .server import MockServer, ServerMode
from .llm_openai import OpenAIStubClient


@click.group()
def main():
    """🐙 Mocktopus - Multi-armed mocks for LLM apps"""


@main.command("serve")
@click.option("--scenario", "-s", type=click.Path(exists=True), help="YAML scenario file")
@click.option("--port", "-p", default=8080, show_default=True, help="Server port")
@click.option("--host", default="127.0.0.1", show_default=True, help="Server host")
@click.option("--mode", type=click.Choice(["mock", "record", "replay"]), default="mock", show_default=True,
              help="Server mode")
@click.option("--recordings-dir", type=click.Path(), help="Directory for recordings (record/replay modes)")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for recording mode")
@click.option("--anthropic-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key for recording mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def serve_cmd(scenario: str, port: int, host: str, mode: str, recordings_dir: str,
              openai_key: str, anthropic_key: str, verbose: bool):
    """
    Start the Mocktopus server to mock LLM APIs.

    Examples:

        # Serve mocks from a scenario file
        mocktopus serve -s examples/haiku.yaml

        # Record real API calls (requires API key)
        mocktopus serve --mode record --recordings-dir ./recordings --openai-key sk-...

        # Replay recorded calls
        mocktopus serve --mode replay --recordings-dir ./recordings
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    server_mode = ServerMode(mode)
    scenario_obj = None

    if server_mode == ServerMode.MOCK and scenario:
        scenario_obj = load_yaml(scenario)
        click.echo(f"📂 Loaded scenario: {scenario} ({len(scenario_obj.rules)} rules)")
    elif server_mode == ServerMode.MOCK:
        click.echo("⚠️  Warning: No scenario file provided, server will return 404 for all requests", err=True)

    server = MockServer(
        scenario=scenario_obj,
        mode=server_mode,
        recordings_dir=recordings_dir,
        real_openai_key=openai_key,
        real_anthropic_key=anthropic_key,
        host=host,
        port=port
    )

    click.echo(f"🐙 Starting Mocktopus server on http://{host}:{port}")
    click.echo(f"📝 Mode: {mode}")
    click.echo(f"🔌 OpenAI endpoint: http://{host}:{port}/v1/chat/completions")
    click.echo(f"🔌 Anthropic endpoint: http://{host}:{port}/v1/messages")
    click.echo("\nPress Ctrl+C to stop the server")

    try:
        server.run()
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")


@main.command("simulate")
@click.option("--scenario", "-s", "scenario_file", required=True, type=click.Path(exists=True),
              help="YAML scenario file")
@click.option("--model", default="gpt-4o-mini", show_default=True, help="Model to simulate")
@click.option("--prompt", required=True, help="User prompt to simulate")
@click.option("--stream/--no-stream", default=False, show_default=True, help="Stream the response")
def simulate_cmd(scenario_file: str, model: str, prompt: str, stream: bool):
    """
    Simulate an LLM call using a scenario file (without starting a server).

    Example:
        mocktopus simulate -s examples/haiku.yaml --prompt "write a haiku"
    """
    scenario = load_yaml(scenario_file)
    client = OpenAIStubClient(scenario)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

    try:
        out = client.chat.completions.create(model=model, messages=messages, stream=stream)

        if stream:
            for ev in out:
                delta = (ev.choices[0].delta.content or "")
                sys.stdout.write(delta)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            click.echo(out.choices[0].message.content)
    except KeyError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@main.command("validate")
@click.argument("scenario_file", type=click.Path(exists=True))
def validate_cmd(scenario_file: str):
    """
    Validate a scenario YAML file.

    Example:
        mocktopus validate examples/haiku.yaml
    """
    try:
        scenario = load_yaml(scenario_file)
        click.echo(f"✅ Valid scenario file with {len(scenario.rules)} rules")
        for i, rule in enumerate(scenario.rules, 1):
            click.echo(f"  Rule {i}: {rule.type} - matches when: {rule.when}")
    except Exception as e:
        click.echo(f"❌ Invalid scenario file: {e}", err=True)
        sys.exit(1)


@main.command("example")
@click.option("--type", "example_type", type=click.Choice(["basic", "streaming", "tools", "multi-model"]),
              default="basic", help="Type of example to generate")
def example_cmd(example_type: str):
    """
    Generate example scenario files.

    Example:
        mocktopus example --type basic > my-scenario.yaml
    """
    examples = {
        "basic": """version: 1
meta:
  description: Basic chat completion mock

rules:
  - type: llm.openai
    when:
      model: "gpt-*"
      messages_contains: "hello"
    respond:
      content: "Hello! How can I help you today?"
      usage:
        input_tokens: 10
        output_tokens: 8
""",
        "streaming": """version: 1
meta:
  description: Streaming response example

rules:
  - type: llm.openai
    when:
      model: "*"
      messages_regex: "stream|realtime"
    respond:
      content: "This is a streaming response that will be sent chunk by chunk."
      delay_ms: 50
      chunk_size: 5
""",
        "tools": """version: 1
meta:
  description: Function calling example

rules:
  - type: llm.openai
    when:
      model: "gpt-4*"
      messages_contains: "weather"
    respond:
      content: null
      tool_calls:
        - id: "call_abc123"
          type: "function"
          function:
            name: "get_weather"
            arguments: '{"location": "San Francisco"}'
""",
        "multi_model": """version: 1
meta:
  description: Multi-model routing example

rules:
  # GPT-4 response
  - type: llm.openai
    when:
      model: "gpt-4*"
      messages_contains: "explain"
    respond:
      content: "Here's a detailed explanation..."

  # GPT-3.5 response
  - type: llm.openai
    when:
      model: "gpt-3.5*"
      messages_contains: "explain"
    respond:
      content: "Here's a simple explanation..."

  # Claude response
  - type: llm.openai
    when:
      model: "claude-*"
      messages_contains: "explain"
    respond:
      content: "I'll explain this step by step..."

  # Default fallback
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "I understand your request."
"""
    }

    click.echo(examples.get(example_type, examples["basic"]))


if __name__ == "__main__":
    main()