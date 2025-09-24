from __future__ import annotations

from typing import Any
import pytest

from .core import Scenario, load_yaml
from .llm_openai import OpenAIStubClient, patch_openai


@pytest.fixture
def use_mocktopus() -> Any:
    """
    A session-scoped Scenario with helpers. Usage:

        def test_x(use_mocktopus):
            use_mocktopus.load_yaml("examples/haiku.yaml")
            client = use_mocktopus.openai_client()
            ...
    """
    scenario = Scenario()

    class _Helper:
        def __init__(self, s: Scenario):
            self.scenario = s

        def load_yaml(self, path: str) -> Any:
            self.scenario.load_yaml(path)
            return self

        def openai_client(self) -> OpenAIStubClient:
            return OpenAIStubClient(self.scenario)

        def patch_openai(self) -> Any:
            return patch_openai(self.scenario)

    return _Helper(scenario)
