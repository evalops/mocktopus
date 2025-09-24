"""
Recording and replay functionality for Mocktopus
"""

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

import aiohttp
from aiohttp import ClientSession


@dataclass
class RecordedInteraction:
    """A recorded API interaction"""
    timestamp: float
    request_method: str
    request_path: str
    request_headers: Dict[str, str]
    request_body: Dict[str, Any]
    response_status: int
    response_headers: Dict[str, str]
    response_body: Any
    response_time_ms: float
    model: Optional[str] = None
    provider: str = "openai"
    interaction_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.interaction_id:
            # Generate deterministic ID from request
            content = f"{self.request_method}:{self.request_path}:{json.dumps(self.request_body, sort_keys=True)}"
            self.interaction_id = hashlib.sha256(content.encode()).hexdigest()[:12]

    def matches_request(self, method: str, path: str, body: Dict[str, Any],
                        fuzzy: bool = True) -> bool:
        """Check if this recording matches a request"""
        if method != self.request_method or path != self.request_path:
            return False

        if not fuzzy:
            return self.request_body == body

        # Fuzzy matching for chat completions
        if path == "/v1/chat/completions":
            # Match on model and messages content (ignore metadata)
            if self.request_body.get("model") != body.get("model"):
                return False

            recorded_messages = self.request_body.get("messages", [])
            request_messages = body.get("messages", [])

            if len(recorded_messages) != len(request_messages):
                return False

            for rec_msg, req_msg in zip(recorded_messages, request_messages):
                if rec_msg.get("role") != req_msg.get("role"):
                    return False
                if rec_msg.get("content") != req_msg.get("content"):
                    return False

            return True

        # Default to exact match for other endpoints
        return self.request_body == body

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedInteraction":
        return cls(**data)


class Recorder:
    """Records API interactions to disk"""

    def __init__(self, recordings_dir: Path):
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.recordings_dir / f"session_{int(time.time())}.jsonl"
        self.interactions: List[RecordedInteraction] = []

    async def record_interaction(self, interaction: RecordedInteraction) -> None:
        """Record a single interaction"""
        self.interactions.append(interaction)

        # Append to file immediately
        with open(self.session_file, "a") as f:
            f.write(json.dumps(interaction.to_dict()) + "\n")

    async def proxy_request(self, method: str, url: str, headers: Dict[str, str],
                           body: Optional[Dict[str, Any]], api_key: str) -> Tuple[int, Dict[str, str], Any]:
        """Proxy a request to the real API and record it"""
        start_time = time.time()

        # Add real API key
        headers = headers.copy()
        headers["Authorization"] = f"Bearer {api_key}"

        async with ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=body if body else None
            ) as response:
                response_body = await response.json()
                response_time_ms = (time.time() - start_time) * 1000

                # Extract model from request
                model = None
                if body:
                    model = body.get("model")

                # Determine provider from URL
                provider = "openai"
                if "anthropic" in url:
                    provider = "anthropic"

                # Record the interaction
                interaction = RecordedInteraction(
                    timestamp=time.time(),
                    request_method=method,
                    request_path=url.replace("https://api.openai.com", "").replace("https://api.anthropic.com", ""),
                    request_headers={k: v for k, v in headers.items() if k.lower() != "authorization"},
                    request_body=body or {},
                    response_status=response.status,
                    response_headers=dict(response.headers),
                    response_body=response_body,
                    response_time_ms=response_time_ms,
                    model=model,
                    provider=provider
                )

                await self.record_interaction(interaction)

                return response.status, dict(response.headers), response_body


class Replayer:
    """Replays recorded API interactions"""

    def __init__(self, recordings_dir: Path):
        self.recordings_dir = Path(recordings_dir)
        self.interactions: List[RecordedInteraction] = []
        self._load_recordings()

    def _load_recordings(self) -> None:
        """Load all recordings from directory"""
        for file in self.recordings_dir.glob("*.jsonl"):
            with open(file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.interactions.append(RecordedInteraction.from_dict(data))

        # Sort by timestamp for chronological replay
        self.interactions.sort(key=lambda x: x.timestamp)

    def find_matching_interaction(self, method: str, path: str, body: Dict[str, Any]) -> Optional[RecordedInteraction]:
        """Find a recorded interaction matching the request"""
        for interaction in self.interactions:
            if interaction.matches_request(method, path, body):
                return interaction
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded interactions"""
        if not self.interactions:
            return {"total": 0}

        stats: Dict[str, Any] = {
            "total": len(self.interactions),
            "by_model": {},
            "by_provider": {},
            "by_endpoint": {},
            "total_response_time_ms": 0,
            "date_range": {
                "first": datetime.fromtimestamp(self.interactions[0].timestamp).isoformat(),
                "last": datetime.fromtimestamp(self.interactions[-1].timestamp).isoformat()
            }
        }

        for interaction in self.interactions:
            # By model
            if interaction.model:
                stats["by_model"][interaction.model] = stats["by_model"].get(interaction.model, 0) + 1

            # By provider
            stats["by_provider"][interaction.provider] = stats["by_provider"].get(interaction.provider, 0) + 1

            # By endpoint
            stats["by_endpoint"][interaction.request_path] = stats["by_endpoint"].get(interaction.request_path, 0) + 1

            # Total response time
            stats["total_response_time_ms"] += interaction.response_time_ms

        return stats