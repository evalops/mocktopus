from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


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
