from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

# JSON Schema for scenario validation
SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1,
            "description": "Schema version (currently only version 1 is supported)"
        },
        "meta": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "author": {"type": "string"},
                "created": {"type": "string"},
                "environment": {"type": "string"},
                "use_cases": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "additionalProperties": True
        },
        "rules": {
            "type": "array",
            "minItems": 0,
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["llm.openai", "llm", "embeddings"],
                        "description": "Rule type"
                    },
                    "when": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "Model pattern (supports wildcards like 'gpt-*')"
                            },
                            "messages_contains": {
                                "type": "string",
                                "description": "Text that must be present in user messages"
                            },
                            "messages_regex": {
                                "type": "string",
                                "description": "Regex pattern for message content"
                            },
                            "endpoint": {
                                "type": "string",
                                "description": "Specific API endpoint to match (e.g., '/v1/embeddings')"
                            }
                        },
                        "additionalProperties": True
                    },
                    "respond": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": ["string", "null"],
                                "description": "Response text content"
                            },
                            "tool_calls": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "type": {"type": "string", "enum": ["function"]},
                                        "function": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "arguments": {"type": "string"}
                                            },
                                            "required": ["name", "arguments"]
                                        }
                                    },
                                    "required": ["id", "type", "function"]
                                }
                            },
                            "usage": {
                                "type": "object",
                                "properties": {
                                    "input_tokens": {"type": "integer", "minimum": 0},
                                    "output_tokens": {"type": "integer", "minimum": 0},
                                    "total_tokens": {"type": "integer", "minimum": 0}
                                },
                                "additionalProperties": False
                            },
                            "embeddings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "embedding": {
                                            "type": "array",
                                            "items": {"type": "number"}
                                        }
                                    },
                                    "required": ["embedding"]
                                }
                            },
                            "images": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string"},
                                        "revised_prompt": {"type": "string"}
                                    },
                                    "required": ["url"]
                                }
                            },
                            "audio_url": {
                                "type": "string",
                                "description": "URL to generated audio file"
                            },
                            "delay_ms": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Delay before responding (milliseconds)"
                            },
                            "chunk_size": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Streaming chunk size in characters"
                            },
                            "id": {
                                "type": "string",
                                "description": "Custom response ID"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional response metadata"
                            }
                        },
                        "additionalProperties": True
                    },
                    "error": {
                        "type": "object",
                        "properties": {
                            "error_type": {
                                "type": "string",
                                "enum": ["rate_limit", "authentication", "timeout", "content_filter", "server_error"],
                                "description": "Type of error to simulate"
                            },
                            "message": {
                                "type": "string",
                                "description": "Error message"
                            },
                            "status_code": {
                                "type": "integer",
                                "minimum": 400,
                                "maximum": 599,
                                "description": "HTTP status code"
                            },
                            "code": {
                                "type": "string",
                                "description": "API-specific error code"
                            },
                            "retry_after": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Retry-After header value in seconds"
                            },
                            "delay_ms": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Delay before error response (milliseconds)"
                            }
                        },
                        "required": ["error_type", "message"],
                        "additionalProperties": False
                    },
                    "times": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Maximum number of times this rule can be used (null = unlimited)"
                    }
                },
                "required": ["type", "when"],
                "additionalProperties": False,
                "anyOf": [
                    {"required": ["respond"]},
                    {"required": ["error"]}
                ]
            }
        }
    },
    "required": ["version", "rules"],
    "additionalProperties": False
}


def validate_scenario_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate scenario data against schema.
    Returns list of validation errors (empty if valid).
    """
    errors = []

    try:
        # Try to import jsonschema for detailed validation
        import jsonschema
        validator = jsonschema.Draft7Validator(SCENARIO_SCHEMA)
        for error in validator.iter_errors(data):
            # Create readable error messages
            path = " -> ".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{path}: {error.message}")
        return errors
    except ImportError:
        # Fallback to basic validation without jsonschema
        pass

    # Basic validation without jsonschema
    if not isinstance(data, dict):
        errors.append("Root must be an object")
        return errors

    # Check version
    if "version" not in data:
        errors.append("version: Required field missing")
    elif not isinstance(data["version"], int) or data["version"] != 1:
        errors.append("version: Must be integer 1")

    # Check rules
    if "rules" not in data:
        errors.append("rules: Required field missing")
    elif not isinstance(data["rules"], list):
        errors.append("rules: Must be an array")
    else:
        for i, rule in enumerate(data["rules"]):
            rule_path = f"rules[{i}]"
            if not isinstance(rule, dict):
                errors.append(f"{rule_path}: Must be an object")
                continue

            # Check required fields
            if "type" not in rule:
                errors.append(f"{rule_path}.type: Required field missing")
            elif rule["type"] not in ["llm.openai", "llm", "embeddings"]:
                errors.append(f"{rule_path}.type: Must be 'llm.openai', 'llm', or 'embeddings'")

            if "when" not in rule:
                errors.append(f"{rule_path}.when: Required field missing")
            elif not isinstance(rule["when"], dict):
                errors.append(f"{rule_path}.when: Must be an object")

            # Must have either respond or error
            has_respond = "respond" in rule
            has_error = "error" in rule
            if not has_respond and not has_error:
                errors.append(f"{rule_path}: Must have either 'respond' or 'error' field")

            # Validate respond structure if present
            if has_respond and not isinstance(rule["respond"], dict):
                errors.append(f"{rule_path}.respond: Must be an object")

            # Validate error structure if present
            if has_error:
                if not isinstance(rule["error"], dict):
                    errors.append(f"{rule_path}.error: Must be an object")
                else:
                    error_obj = rule["error"]
                    if "error_type" not in error_obj:
                        errors.append(f"{rule_path}.error.error_type: Required field missing")
                    if "message" not in error_obj:
                        errors.append(f"{rule_path}.error.message: Required field missing")

            # Validate times if present
            if "times" in rule and rule["times"] is not None:
                if not isinstance(rule["times"], int) or rule["times"] < 1:
                    errors.append(f"{rule_path}.times: Must be positive integer or null")

    # Check meta if present
    if "meta" in data and not isinstance(data["meta"], dict):
        errors.append("meta: Must be an object")

    return errors


@dataclass
class Rule:
    """A single matching/responding rule for LLM calls."""

    type: str  # e.g., "llm.openai"
    when: Dict[str, Any]
    respond: Dict[str, Any]
    times: Optional[int] = None  # None => unlimited
    error: Optional[Dict[str, Any]] = None  # Error response config
    _remaining: Optional[int] = field(default=None, init=False, repr=False)

    def ok_to_use(self) -> bool:
        if self.times is None:
            return True
        if self._remaining is None:
            self._remaining = self.times
        return self._remaining > 0

    def consume(self) -> None:
        if self.times is None:
            return
        if self._remaining is None:
            self._remaining = self.times
        if self._remaining > 0:
            self._remaining -= 1


class Scenario:
    """
    Holds a set of rules and finds the best match for a given invocation.
    Focuses on LLM chat completions (OpenAI‑style) for the MVP.
    """

    def __init__(self, rules: Optional[List[Rule]] = None, meta: Optional[Dict[str, Any]] = None):
        self.rules: List[Rule] = rules or []
        self.meta: Dict[str, Any] = meta or {}

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    # --- Matching helpers -------------------------------------------------

    def _extract_last_user_text(self, messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):  # OpenAI content blocks
                    # take text parts
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join([t for t in text_parts if t])
        return ""

    def _llm_rule_matches(self, rule: Rule, *, model: str, messages: List[Dict[str, Any]]) -> bool:
        if rule.type not in ("llm.openai", "llm"):
            return False
        if not rule.ok_to_use():
            return False

        cond = rule.when or {}
        # model glob match
        model_glob = cond.get("model", "*")
        if model_glob and not fnmatch.fnmatch(model or "", model_glob):
            return False

        # simple substring match on last user message
        sub = cond.get("messages_contains")
        if sub:
            user_text = self._extract_last_user_text(messages)
            if sub not in user_text:
                return False

        # regex on concatenated message text
        rx = cond.get("messages_regex")
        if rx:
            try:
                pattern = re.compile(rx, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            except re.error:
                return False
            all_text = " ".join(
                [
                    m.get("content", "") if isinstance(m.get("content", ""), str) else ""
                    for m in messages
                ]
            )
            if not pattern.search(all_text):
                return False

        return True

    def find_llm(self, *, model: str, messages: List[Dict[str, Any]]) -> Tuple[Optional[Rule], Optional[Dict[str, Any]]]:
        for rule in self.rules:
            if self._llm_rule_matches(rule, model=model, messages=messages):
                # Return error config if present, otherwise normal response
                response_config = rule.error if rule.error else rule.respond
                return rule, response_config
        return None, None

    # --- YAML I/O ---------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Scenario":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Validate schema
        validation_errors = validate_scenario_data(data)
        if validation_errors:
            error_msg = f"Schema validation failed for {path}:\n"
            for i, error in enumerate(validation_errors, 1):
                error_msg += f"  {i}. {error}\n"
            error_msg += "\n💡 Use 'mocktopus validate' for detailed validation help"
            raise ValueError(error_msg)

        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported fixture version: {version}")
        rules_data = data.get("rules", [])
        rules = []
        for r in rules_data:
            rules.append(
                Rule(
                    type=r.get("type", "llm.openai"),
                    when=r.get("when", {}) or {},
                    respond=r.get("respond", {}) or {},
                    times=r.get("times"),
                    error=r.get("error"),
                )
            )
        meta = data.get("meta", {})
        return cls(rules=rules, meta=meta)

    def to_yaml(self) -> str:
        data = {
            "version": 1,
            "meta": self.meta,
            "rules": [
                {"type": r.type, "when": r.when, "respond": r.respond, "times": r.times}
                for r in self.rules
            ],
        }
        return yaml.safe_dump(data, sort_keys=False)

    # Convenience API for pytest fixture
    def load_yaml(self, path: str) -> "Scenario":
        other = Scenario.from_yaml(path)
        # merge
        self.rules.extend(other.rules)
        self.meta.update(other.meta)
        return self


def load_yaml(path: str) -> Scenario:
    return Scenario.from_yaml(path)
