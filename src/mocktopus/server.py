"""
Mocktopus LLM API Mock Server

Drop-in replacement for OpenAI/Anthropic APIs for deterministic testing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import yaml
from aiohttp import web
from aiohttp.web import Request, Response, StreamResponse

from .core import Scenario

logger = logging.getLogger(__name__)


class ServerMode(Enum):
    MOCK = "mock"
    RECORD = "record"
    REPLAY = "replay"


@dataclass
class MockServer:
    """
    LLM API Mock Server that mimics OpenAI/Anthropic endpoints.

    Modes:
    - mock: Use predefined scenarios from YAML
    - record: Proxy to real API and save interactions
    - replay: Serve previously recorded interactions
    """

    scenario: Optional[Scenario] = None
    mode: ServerMode = ServerMode.MOCK
    recordings_dir: Optional[Path] = None
    real_openai_key: Optional[str] = None
    real_anthropic_key: Optional[str] = None
    port: int = 8080
    host: str = "127.0.0.1"

    def __post_init__(self):
        self.recordings = []
        self.replay_index = 0
        if self.recordings_dir:
            self.recordings_dir = Path(self.recordings_dir)
            self.recordings_dir.mkdir(parents=True, exist_ok=True)

    # OpenAI Chat Completions Handler
    async def handle_openai_chat(self, request: Request) -> Union[Response, StreamResponse]:
        """Handle /v1/chat/completions endpoint (OpenAI-compatible)"""

        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
                status=400
            )

        model = data.get("model", "gpt-3.5-turbo")
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        temperature = data.get("temperature", 1.0)
        max_tokens = data.get("max_tokens")
        tools = data.get("tools", [])
        tool_choice = data.get("tool_choice")

        # Mode-specific handling
        if self.mode == ServerMode.MOCK:
            return await self._handle_mock_openai(model, messages, stream, data)
        elif self.mode == ServerMode.RECORD:
            return await self._handle_record_openai(data, stream)
        elif self.mode == ServerMode.REPLAY:
            return await self._handle_replay_openai(data, stream)

    async def _handle_mock_openai(self, model: str, messages: List[Dict],
                                  stream: bool, full_request: Dict) -> Union[Response, StreamResponse]:
        """Handle mocked OpenAI responses using scenarios"""

        if not self.scenario:
            return web.json_response(
                {"error": {"message": "No scenario loaded", "type": "server_error"}},
                status=500
            )

        # Find matching rule
        rule, respond_config = self.scenario.find_llm(model=model, messages=messages)

        if not rule:
            return web.json_response(
                {"error": {"message": "No matching mock rule found", "type": "not_found"}},
                status=404
            )

        rule.consume()

        # Extract response config
        content = respond_config.get("content", "Mocked response")
        delay_ms = respond_config.get("delay_ms", 0)
        tool_calls = respond_config.get("tool_calls", [])
        usage = respond_config.get("usage", {})

        # Handle delay
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

        # Stream response
        if stream:
            return await self._stream_openai_response(content, model, tool_calls)

        # Regular response
        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 10),
                "completion_tokens": usage.get("output_tokens", 20),
                "total_tokens": usage.get("total_tokens", 30)
            },
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        }

        # Add tool calls if present
        if tool_calls:
            response_data["choices"][0]["message"]["tool_calls"] = tool_calls
            response_data["choices"][0]["message"]["content"] = None
            response_data["choices"][0]["finish_reason"] = "tool_calls"

        return web.json_response(response_data)

    async def _stream_openai_response(self, content: str, model: str,
                                      tool_calls: List[Dict] = None) -> StreamResponse:
        """Stream OpenAI response using Server-Sent Events"""

        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'

        await response.prepare()

        # Stream ID
        stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Stream content chunks
        chunk_size = 5  # characters per chunk
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]

            chunk_data = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None
                }]
            }

            await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
            await asyncio.sleep(0.02)  # Simulate token delay

        # Send finish chunk
        finish_data = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }

        await response.write(f"data: {json.dumps(finish_data)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")

        return response

    async def _handle_record_openai(self, request_data: Dict, stream: bool) -> Response:
        """Record real OpenAI API calls"""

        if not self.real_openai_key:
            return web.json_response(
                {"error": {"message": "OpenAI API key not configured for recording", "type": "server_error"}},
                status=500
            )

        # TODO: Implement actual OpenAI API proxy and recording
        # This would use aiohttp to call real OpenAI API and save the interaction

        return web.json_response(
            {"error": {"message": "Recording mode not yet implemented", "type": "not_implemented"}},
            status=501
        )

    async def _handle_replay_openai(self, request_data: Dict, stream: bool) -> Response:
        """Replay recorded OpenAI interactions"""

        if not self.recordings:
            return web.json_response(
                {"error": {"message": "No recordings available", "type": "not_found"}},
                status=404
            )

        # TODO: Implement replay from recordings

        return web.json_response(
            {"error": {"message": "Replay mode not yet implemented", "type": "not_implemented"}},
            status=501
        )

    # Anthropic Messages Handler
    async def handle_anthropic_messages(self, request: Request) -> Union[Response, StreamResponse]:
        """Handle /v1/messages endpoint (Anthropic-compatible)"""

        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": {"type": "invalid_request_error", "message": f"Invalid JSON: {e}"}},
                status=400
            )

        model = data.get("model", "claude-3-sonnet")
        messages = data.get("messages", [])
        stream = data.get("stream", False)

        # For now, convert to OpenAI format and use same handler
        # TODO: Implement proper Anthropic format handling

        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })

        # Mock response in Anthropic format
        if not self.scenario:
            return web.json_response(
                {"error": {"type": "server_error", "message": "No scenario loaded"}},
                status=500
            )

        rule, respond_config = self.scenario.find_llm(model=model, messages=openai_messages)

        if not rule:
            return web.json_response(
                {"error": {"type": "not_found", "message": "No matching mock rule found"}},
                status=404
            )

        rule.consume()
        content = respond_config.get("content", "Mocked response")

        response_data = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "type": "message",
            "model": model,
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": content
            }],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }

        return web.json_response(response_data)

    # Health check endpoint
    async def handle_health(self, request: Request) -> Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "mode": self.mode.value,
            "scenario_loaded": self.scenario is not None,
            "recordings_count": len(self.recordings)
        })

    # Models endpoint (OpenAI)
    async def handle_models(self, request: Request) -> Response:
        """List available models"""
        return web.json_response({
            "object": "list",
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-4-turbo", "object": "model"},
                {"id": "claude-3-sonnet", "object": "model"},
                {"id": "claude-3-opus", "object": "model"}
            ]
        })

    def create_app(self) -> web.Application:
        """Create the aiohttp application"""
        app = web.Application()

        # OpenAI-compatible endpoints
        app.router.add_post('/v1/chat/completions', self.handle_openai_chat)
        app.router.add_get('/v1/models', self.handle_models)

        # Anthropic-compatible endpoints
        app.router.add_post('/v1/messages', self.handle_anthropic_messages)

        # Health check
        app.router.add_get('/health', self.handle_health)

        return app

    def run(self):
        """Run the mock server"""
        app = self.create_app()

        logger.info(f"🐙 Mocktopus server starting on http://{self.host}:{self.port}")
        logger.info(f"Mode: {self.mode.value}")
        if self.scenario:
            logger.info(f"Scenario loaded with {len(self.scenario.rules)} rules")

        web.run_app(app, host=self.host, port=self.port, print=False)