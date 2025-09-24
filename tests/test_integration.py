"""
End-to-end integration tests for Mocktopus
"""

import asyncio
import json
import pytest
import aiohttp
from pathlib import Path
import tempfile
import time

from mocktopus import Scenario, load_yaml
from mocktopus.server import MockServer, ServerMode
from mocktopus.cost_tracker import CostTracker
from mocktopus.core import Rule


@pytest.fixture
async def mock_server():
    """Start a mock server for testing"""
    scenario = Scenario()
    scenario.rules.append(Rule(
        type="llm.openai",
        when={"messages_contains": "hello"},
        respond={"content": "Hello from mock!", "usage": {"input_tokens": 5, "output_tokens": 4}}
    ))

    # Use port 0 to get a random available port
    server = MockServer(scenario=scenario, port=0, host="127.0.0.1")
    app = server.create_app()

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Update server port with the actual port that was bound
    server.port = site._server.sockets[0].getsockname()[1]

    yield server

    await runner.cleanup()


@pytest.mark.asyncio
async def test_basic_chat_completion(mock_server):
    """Test basic chat completion endpoint"""
    async with aiohttp.ClientSession() as session:
        # Test successful request
        async with session.post(
            f"http://localhost:{mock_server.port}/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}]
            }
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "choices" in data
            assert data["choices"][0]["message"]["content"] == "Hello from mock!"

        # Test no match
        async with session.post(
            f"http://localhost:{mock_server.port}/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "unknown"}]
            }
        ) as resp:
            assert resp.status == 404


@pytest.mark.asyncio
async def test_streaming_response():
    """Test SSE streaming response"""
    scenario = load_yaml("examples/chat-basic.yaml")
    server = MockServer(scenario=scenario, port=0)
    app = server.create_app()

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Update server port with the actual port that was bound
    server.port = site._server.sockets[0].getsockname()[1]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{server.port}/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True
                }
            ) as resp:
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "text/event-stream"

                chunks = []
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data != "[DONE]":
                            chunks.append(json.loads(data))

                assert len(chunks) > 0
                assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cost_tracking():
    """Test cost tracking functionality"""
    tracker = CostTracker()

    # Track some requests
    cost1 = tracker.track("gpt-4", 100, 200)
    assert cost1 > 0

    cost2 = tracker.track("gpt-3.5-turbo", 500, 1000)
    assert cost2 > 0

    report = tracker.get_report()
    assert report.requests_mocked == 2
    assert report.total_saved == cost1 + cost2
    assert "gpt-4" in report.breakdown_by_model
    assert "gpt-3.5-turbo" in report.breakdown_by_model

    # Test summary generation
    summary = report.get_summary()
    assert "Cost Savings Report" in summary
    assert "$" in summary


@pytest.mark.asyncio
async def test_error_scenarios():
    """Test error response mocking"""
    scenario = load_yaml("examples/errors.yaml")
    server = MockServer(scenario=scenario, port=8091)
    app = server.create_app()

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 8091)
    await site.start()

    try:
        async with aiohttp.ClientSession() as session:
            # Test rate limit error
            async with session.post(
                "http://localhost:8091/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "rate limit test"}]
                }
            ) as resp:
                assert resp.status == 429
                assert "Retry-After" in resp.headers
                data = await resp.json()
                assert "error" in data
                assert data["error"]["type"] == "rate_limit_error"

            # Test auth error
            async with session.post(
                "http://localhost:8091/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "auth test"}]
                }
            ) as resp:
                assert resp.status == 401
                data = await resp.json()
                assert data["error"]["type"] == "authentication_error"

            # Test retry scenario (fails twice, then succeeds)
            for i in range(3):
                async with session.post(
                    "http://localhost:8091/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "retry test"}]
                    }
                ) as resp:
                    if i < 2:
                        assert resp.status == 503
                        data = await resp.json()
                        assert "error" in data
                    else:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["choices"][0]["message"]["content"] == "Success after retries!"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint"""
    server = MockServer(port=8092)
    app = server.create_app()

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 8092)
    await site.start()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8092/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"
                assert data["mode"] == "mock"
                assert "cost_saved" in data
                assert "requests_mocked" in data
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cost_report_endpoint():
    """Test cost report endpoint"""
    scenario = load_yaml("examples/chat-basic.yaml")
    server = MockServer(scenario=scenario, port=8093)
    app = server.create_app()

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 8093)
    await site.start()

    try:
        async with aiohttp.ClientSession() as session:
            # Make some requests to track costs
            for _ in range(3):
                await session.post(
                    "http://localhost:8093/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hello"}]
                    }
                )

            # Get cost report
            async with session.get("http://localhost:8093/cost-report") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "report" in data
                assert "summary" in data
                assert data["report"]["requests_mocked"] == 3
                assert data["report"]["total_saved"] > 0
    finally:
        await runner.cleanup()


def test_scenario_loading():
    """Test loading scenarios from YAML files"""
    # Test basic scenario
    scenario = load_yaml("examples/chat-basic.yaml")
    assert len(scenario.rules) > 0

    # Test error scenario
    error_scenario = load_yaml("examples/errors.yaml")
    assert len(error_scenario.rules) > 0

    # Check error rules have error field
    has_error_rule = any(r.error is not None for r in error_scenario.rules)
    assert has_error_rule

    # Test tool calling scenario
    tool_scenario = load_yaml("examples/tool-calling.yaml")
    assert len(tool_scenario.rules) > 0


def test_rule_matching():
    """Test rule matching logic"""
    scenario = Scenario()

    # Test exact match
    scenario.rules.append(Rule(
        type="llm.openai",
        when={"messages_contains": "weather"},
        respond={"content": "It's sunny!"}
    ))

    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather?"}]
    )
    assert rule is not None
    assert response["content"] == "It's sunny!"

    # Test regex match
    scenario.rules = []
    scenario.rules.append(Rule(
        type="llm.openai",
        when={"messages_regex": r"\d+ \+ \d+"},
        respond={"content": "Math detected"}
    ))

    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is 2 + 2?"}]
    )
    assert rule is not None
    assert response["content"] == "Math detected"

    # Test model glob match
    scenario.rules = []
    scenario.rules.append(Rule(
        type="llm.openai",
        when={"model": "gpt-3.5*"},
        respond={"content": "GPT-3.5 response"}
    ))

    rule, response = scenario.find_llm(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert rule is not None
    assert response["content"] == "GPT-3.5 response"

    # No match for different model
    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert rule is None


def test_usage_limits():
    """Test rule usage limits"""
    scenario = Scenario()

    rule = Rule(
        type="llm.openai",
        when={"messages_contains": "test"},
        respond={"content": "Limited"},
        times=2
    )
    scenario.rules.append(rule)

    # Should work twice
    for i in range(2):
        matched, _ = scenario.find_llm(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )
        assert matched is not None
        matched.consume()

    # Should not work third time
    matched, _ = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}]
    )
    assert matched is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])