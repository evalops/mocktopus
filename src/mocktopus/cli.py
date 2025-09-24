from __future__ import annotations

import sys
from typing import List, Dict, Any
import click

from .core import Scenario, load_yaml
from .llm_openai import OpenAIStubClient


@click.group()
def main():
    """Mocktopus CLI."""


@main.command("simulate")
@click.option("--fixture", "-f", "fixture", required=True, type=click.Path(exists=True))
@click.option("--model", default="gpt-4o-mini", show_default=True)
@click.option("--prompt", required=True, help="User prompt to simulate.")
@click.option("--stream/--no-stream", default=False, show_default=True)
def simulate_cmd(fixture: str, model: str, prompt: str, stream: bool):
    """Simulate an LLM call using a fixture file."""
    scenario = load_yaml(fixture)
    client = OpenAIStubClient(scenario)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    out = client.chat.completions.create(model=model, messages=messages, stream=stream)
    if stream:
        for ev in out:
            delta = (ev.choices[0].delta.content or "")
            sys.stdout.write(delta)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        click.echo(out.choices[0].message.content)
