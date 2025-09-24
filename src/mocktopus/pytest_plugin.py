from __future__ import annotations

import pytest

from .core import Scenario, load_yaml
from .llm_openai import OpenAIStubClient, patch_openai


@pytest.fixture
def use_mocktopus():
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

        def load_yaml(self, path: str):
            self.scenario.load_yaml(path)
            return self

        def openai_client(self) -> OpenAIStubClient:
            return OpenAIStubClient(self.scenario)

        def patch_openai(self):
            return patch_openai(self.scenario)

    return _Helper(scenario)
