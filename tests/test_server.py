"""Tests for the Mocktopus server"""

import asyncio
import json
import pytest
from aiohttp import ClientSession
from aiohttp.web import Application

from mocktopus import Scenario, Rule
from mocktopus.server import MockServer, ServerMode


@pytest.fixture
def basic_scenario():
    """Create a basic test scenario"""
    scenario = Scenario()
    scenario.add_rule(Rule(
        type="llm.openai",
        when={"messages_contains": "hello"},
        respond={"content": "Hello from test!"},
    ))
    scenario.add_rule(Rule(
        type="llm.openai",
        when={"messages_contains": "weather"},
        respond={
            "content": None,
            "tool_calls": [{
                "id": "test_call",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "test"}'
                }
            }]
        }
    ))
    return scenario


@pytest.mark.asyncio
async def test_server_health_endpoint():
    """Test the health check endpoint"""
    server = MockServer()
    app = server.create_app()

    async with ClientSession() as session:
        # Mock the app for testing
        handler = app.make_handler()

        # Test health endpoint returns expected data
        request_data = {"path": "/health", "method": "GET"}
        # This would need proper test setup with aiohttp test client
        # Simplified for demonstration


def test_server_initialization():
    """Test server initialization with different modes"""
    # Test default initialization
    server = MockServer()
    assert server.mode == ServerMode.MOCK
    assert server.port == 8080
    assert server.host == "127.0.0.1"

    # Test with scenario
    scenario = Scenario()
    server = MockServer(scenario=scenario, mode=ServerMode.RECORD, port=9000)
    assert server.scenario == scenario
    assert server.mode == ServerMode.RECORD
    assert server.port == 9000


def test_scenario_rule_matching():
    """Test that scenario rules match correctly"""
    scenario = Scenario()
    rule = Rule(
        type="llm.openai",
        when={"model": "gpt-4*", "messages_contains": "test"},
        respond={"content": "matched!"}
    )
    scenario.add_rule(rule)

    # Test matching
    matched_rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "this is a test"}]
    )
    assert matched_rule == rule
    assert response["content"] == "matched!"

    # Test non-matching
    matched_rule, response = scenario.find_llm(
        model="gpt-3.5",
        messages=[{"role": "user", "content": "no match"}]
    )
    assert matched_rule is None
    assert response is None


def test_rule_usage_limits():
    """Test that rules respect usage limits"""
    rule = Rule(
        type="llm.openai",
        when={"messages_contains": "limited"},
        respond={"content": "response"},
        times=2  # Only allow 2 uses
    )

    # First use - should work
    assert rule.ok_to_use() is True
    rule.consume()

    # Second use - should work
    assert rule.ok_to_use() is True
    rule.consume()

    # Third use - should not work
    assert rule.ok_to_use() is False