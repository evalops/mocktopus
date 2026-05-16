"""End-to-end integration tests for Mocktopus."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from mocktopus import Scenario, load_yaml
from mocktopus.core import Rule
from mocktopus.cost_tracker import CostTracker
from mocktopus.server import MockServer


@asynccontextmanager
async def run_server(server: MockServer) -> AsyncIterator[MockServer]:
    app = server.create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, server.host, 0)
    await site.start()

    sockets = site._server.sockets if site._server else None
    assert sockets
    server.port = sockets[0].getsockname()[1]

    try:
        yield server
    finally:
        await runner.cleanup()


def url(server: MockServer, path: str) -> str:
    return f"http://{server.host}:{server.port}{path}"


@pytest.fixture
def hello_scenario() -> Scenario:
    scenario = Scenario()
    scenario.rules.append(
        Rule(
            type="llm.openai",
            when={"messages_contains": "hello"},
            respond={
                "content": "Hello from mock!",
                "usage": {"input_tokens": 5, "output_tokens": 4},
            },
        )
    )
    return scenario


@pytest.mark.asyncio
async def test_basic_chat_completion(hello_scenario: Scenario) -> None:
    async with run_server(MockServer(scenario=hello_scenario, port=0)) as server:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url(server, "/v1/chat/completions"),
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["choices"][0]["message"]["content"] == "Hello from mock!"

            async with session.post(
                url(server, "/v1/chat/completions"),
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "unknown"}]},
            ) as resp:
                assert resp.status == 404


@pytest.mark.asyncio
async def test_streaming_response() -> None:
    scenario = load_yaml("examples/chat-basic.yaml")

    async with run_server(MockServer(scenario=scenario, port=0)) as server:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url(server, "/v1/chat/completions"),
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            ) as resp:
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "text/event-stream"

                chunks = []
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data != "[DONE]":
                            chunks.append(json.loads(data))

                assert chunks
                assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_cost_tracking() -> None:
    tracker = CostTracker()

    cost1 = tracker.track("gpt-4", 100, 200)
    assert cost1 > 0

    cost2 = tracker.track("gpt-3.5-turbo", 500, 1000)
    assert cost2 > 0

    report = tracker.get_report()
    assert report.requests_mocked == 2
    assert report.total_saved == cost1 + cost2
    assert "gpt-4" in report.breakdown_by_model
    assert "gpt-3.5-turbo" in report.breakdown_by_model

    summary = report.get_summary()
    assert "Cost Savings Report" in summary
    assert "$" in summary


@pytest.mark.asyncio
async def test_error_scenarios() -> None:
    scenario = load_yaml("examples/errors.yaml")

    async with run_server(MockServer(scenario=scenario, port=0)) as server:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url(server, "/v1/chat/completions"),
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "rate limit test"}],
                },
            ) as resp:
                assert resp.status == 429
                assert "Retry-After" in resp.headers
                data = await resp.json()
                assert data["error"]["type"] == "rate_limit_error"

            async with session.post(
                url(server, "/v1/chat/completions"),
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "auth test"}],
                },
            ) as resp:
                assert resp.status == 401
                data = await resp.json()
                assert data["error"]["type"] == "authentication_error"

            async with session.post(
                url(server, "/v1/chat/completions"),
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "invalid request"}],
                },
            ) as resp:
                assert resp.status == 400
                data = await resp.json()
                assert data["error"]["type"] == "invalid_request_error"
                assert "gpt-5" in data["error"]["message"]

            for i in range(3):
                async with session.post(
                    url(server, "/v1/chat/completions"),
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "retry test"}],
                    },
                ) as resp:
                    if i < 2:
                        assert resp.status == 503
                        data = await resp.json()
                        assert "error" in data
                    else:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["choices"][0]["message"]["content"] == "Success after retries!"


@pytest.mark.asyncio
async def test_embeddings_scenario_matches_endpoint_rule() -> None:
    scenario = Scenario(
        rules=[
            Rule(
                type="embeddings",
                when={"endpoint": "/v1/embeddings"},
                respond={
                    "embeddings": [{"embedding": [0.1, 0.2, -0.3]}],
                    "usage": {"input_tokens": 7, "total_tokens": 7},
                },
            )
        ]
    )

    async with run_server(MockServer(scenario=scenario, port=0)) as server:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url(server, "/v1/embeddings"),
                json={"model": "text-embedding-3-small", "input": "hello embeddings"},
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["data"][0]["embedding"] == [0.1, 0.2, -0.3]
                assert data["usage"]["prompt_tokens"] == 7


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    async with run_server(MockServer(port=0)) as server:
        async with aiohttp.ClientSession() as session:
            async with session.get(url(server, "/health")) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"
                assert data["mode"] == "mock"
                assert "cost_saved" in data
                assert "requests_mocked" in data


@pytest.mark.asyncio
async def test_cost_report_endpoint() -> None:
    scenario = load_yaml("examples/chat-basic.yaml")

    async with run_server(MockServer(scenario=scenario, port=0)) as server:
        async with aiohttp.ClientSession() as session:
            for _ in range(3):
                await session.post(
                    url(server, "/v1/chat/completions"),
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
                )

            async with session.get(url(server, "/cost-report")) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "report" in data
                assert "summary" in data
                assert data["report"]["requests_mocked"] == 3
                assert data["report"]["total_saved"] > 0


def test_scenario_loading() -> None:
    scenario = load_yaml("examples/chat-basic.yaml")
    assert scenario.rules

    error_scenario = load_yaml("examples/errors.yaml")
    assert any(rule.error is not None for rule in error_scenario.rules)

    tool_scenario = load_yaml("examples/tool-calling.yaml")
    assert tool_scenario.rules


def test_rule_matching() -> None:
    scenario = Scenario()

    scenario.rules.append(
        Rule(
            type="llm.openai",
            when={"messages_contains": "weather"},
            respond={"content": "It's sunny!"},
        )
    )

    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather?"}],
    )
    assert rule is not None
    assert response["content"] == "It's sunny!"

    scenario.rules = [
        Rule(
            type="llm.openai",
            when={"messages_regex": r"\d+ \+ \d+"},
            respond={"content": "Math detected"},
        )
    ]
    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    assert rule is not None
    assert response["content"] == "Math detected"

    scenario.rules = [
        Rule(
            type="llm.openai",
            when={"model": "gpt-3.5*"},
            respond={"content": "GPT-3.5 response"},
        )
    ]
    rule, response = scenario.find_llm(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert rule is not None
    assert response["content"] == "GPT-3.5 response"

    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert rule is None
    assert response is None


def test_usage_limits() -> None:
    scenario = Scenario(
        rules=[
            Rule(
                type="llm.openai",
                when={"messages_contains": "test"},
                respond={"content": "Limited"},
                times=2,
            )
        ]
    )

    for _ in range(2):
        matched, _ = scenario.find_llm(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
        )
        assert matched is not None
        matched.consume()

    matched, _ = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
    )
    assert matched is None
