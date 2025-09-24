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
from .recorder import Recorder, Replayer, RecordedInteraction
from .cost_tracker import CostTracker

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
        self.cost_tracker = CostTracker()
        self.recorder = None
        self.replayer = None

        if self.recordings_dir:
            self.recordings_dir = Path(self.recordings_dir)
            self.recordings_dir.mkdir(parents=True, exist_ok=True)

            if self.mode == ServerMode.RECORD:
                self.recorder = Recorder(self.recordings_dir)
            elif self.mode == ServerMode.REPLAY:
                self.replayer = Replayer(self.recordings_dir)

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
            return await self._handle_mock_openai(request, model, messages, stream, data)
        elif self.mode == ServerMode.RECORD:
            return await self._handle_record_openai(data, stream)
        elif self.mode == ServerMode.REPLAY:
            return await self._handle_replay_openai(data, stream)

    async def _handle_mock_openai(self, request: Request, model: str, messages: List[Dict],
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

        # Check if this is an error response
        if respond_config and "error_type" in respond_config:
            return await self._handle_error_response(respond_config)

        # Extract response config (handle None case)
        if not respond_config:
            respond_config = {}
        content = respond_config.get("content", "Mocked response")
        delay_ms = respond_config.get("delay_ms", 0)
        tool_calls = respond_config.get("tool_calls", [])
        usage = respond_config.get("usage", {})

        # Track cost savings
        input_tokens = usage.get("input_tokens", 100)
        output_tokens = usage.get("output_tokens", 200)
        self.cost_tracker.track(model, input_tokens, output_tokens)

        # Handle delay
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

        # Stream response
        if stream:
            return await self._stream_openai_response(request, content, model, tool_calls)

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

    async def _handle_error_response(self, error_config: Dict[str, Any]) -> Response:
        """Handle error response configuration"""
        error_type = error_config.get("error_type", "server_error")
        message = error_config.get("message", "Mock error")
        status_code = error_config.get("status_code", 500)
        delay_ms = error_config.get("delay_ms", 0)

        # Handle delay
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

        # Build error response based on type
        if error_type == "rate_limit":
            return web.json_response(
                {
                    "error": {
                        "message": message or "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                },
                status=status_code or 429,
                headers={"Retry-After": str(error_config.get("retry_after", 60))}
            )
        elif error_type == "invalid_request":
            return web.json_response(
                {
                    "error": {
                        "message": message or "Invalid request",
                        "type": "invalid_request_error",
                        "code": "invalid_request"
                    }
                },
                status=status_code or 400
            )
        elif error_type == "authentication":
            return web.json_response(
                {
                    "error": {
                        "message": message or "Invalid API key",
                        "type": "authentication_error",
                        "code": "invalid_api_key"
                    }
                },
                status=status_code or 401
            )
        elif error_type == "timeout":
            # Simulate timeout by waiting then returning error
            if delay_ms == 0:
                await asyncio.sleep(30)  # Default 30 second timeout
            return web.json_response(
                {
                    "error": {
                        "message": message or "Request timeout",
                        "type": "timeout_error",
                        "code": "timeout"
                    }
                },
                status=status_code or 504
            )
        else:
            # Generic error
            return web.json_response(
                {
                    "error": {
                        "message": message,
                        "type": error_type,
                        "code": error_config.get("code", "unknown_error")
                    }
                },
                status=status_code
            )

    async def _stream_openai_response(self, request: Request, content: str, model: str,
                                      tool_calls: Optional[List[Dict]] = None) -> StreamResponse:
        """Stream OpenAI response using Server-Sent Events"""

        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'

        await response.prepare(request)

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

        if not self.recorder:
            return web.json_response(
                {"error": {"message": "Recorder not initialized", "type": "server_error"}},
                status=500
            )

        # Proxy to real OpenAI API
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        try:
            status, resp_headers, resp_body = await self.recorder.proxy_request(
                method="POST",
                url=url,
                headers=headers,
                body=request_data,
                api_key=self.real_openai_key
            )

            # Track cost of real API call
            model = request_data.get("model", "gpt-3.5-turbo")
            if "usage" in resp_body:
                self.cost_tracker.track(
                    model,
                    resp_body["usage"].get("prompt_tokens", 0),
                    resp_body["usage"].get("completion_tokens", 0)
                )

            return web.json_response(resp_body, status=status)

        except Exception as e:
            logger.error(f"Error proxying request: {e}")
            return web.json_response(
                {"error": {"message": f"Proxy error: {str(e)}", "type": "proxy_error"}},
                status=500
            )

    async def _handle_replay_openai(self, request_data: Dict, stream: bool) -> Response:
        """Replay recorded OpenAI interactions"""

        if not self.replayer:
            return web.json_response(
                {"error": {"message": "Replayer not initialized", "type": "server_error"}},
                status=500
            )

        # Find matching recording
        interaction = self.replayer.find_matching_interaction(
            method="POST",
            path="/v1/chat/completions",
            body=request_data
        )

        if not interaction:
            return web.json_response(
                {"error": {"message": "No matching recording found", "type": "not_found"}},
                status=404
            )

        # Track cost savings from replay
        model = request_data.get("model", "gpt-3.5-turbo")
        if isinstance(interaction.response_body, dict) and "usage" in interaction.response_body:
            self.cost_tracker.track(
                model,
                interaction.response_body["usage"].get("prompt_tokens", 100),
                interaction.response_body["usage"].get("completion_tokens", 200)
            )

        # Simulate original response time
        if interaction.response_time_ms > 0:
            await asyncio.sleep(interaction.response_time_ms / 1000)

        return web.json_response(interaction.response_body, status=interaction.response_status)

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
        stats = {}
        if self.replayer:
            stats = self.replayer.get_statistics()

        return web.json_response({
            "status": "healthy",
            "mode": self.mode.value,
            "scenario_loaded": self.scenario is not None,
            "recordings_stats": stats,
            "cost_saved": f"${self.cost_tracker.get_report().total_saved:.2f}",
            "requests_mocked": self.cost_tracker.get_report().requests_mocked
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

        # Cost report endpoint
        app.router.add_get('/cost-report', self.handle_cost_report)

        return app

    async def handle_cost_report(self, request: Request) -> Response:
        """Get cost savings report"""
        report = self.cost_tracker.get_report()
        return web.json_response({
            "report": report.to_json(),
            "summary": report.get_summary()
        })

    def run(self):
        """Run the mock server"""
        app = self.create_app()

        logger.info(f"🐙 Mocktopus server starting on http://{self.host}:{self.port}")
        logger.info(f"Mode: {self.mode.value}")
        if self.scenario:
            logger.info(f"Scenario loaded with {len(self.scenario.rules)} rules")
        if self.mode == ServerMode.RECORD:
            logger.info(f"Recording to: {self.recordings_dir}")
        elif self.mode == ServerMode.REPLAY:
            if self.replayer:
                stats = self.replayer.get_statistics()
                logger.info(f"Replaying {stats.get('total', 0)} recorded interactions")

        # Add shutdown handler to show cost report
        async def on_shutdown(app):
            report = self.cost_tracker.get_report()
            if report.requests_mocked > 0:
                print("\n" + report.get_summary())

        app.on_shutdown.append(on_shutdown)

        web.run_app(app, host=self.host, port=self.port, print=False)