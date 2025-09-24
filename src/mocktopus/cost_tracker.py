"""
Cost tracking for Mocktopus - shows how much money you save
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ModelPricing:
    """Pricing for a specific model"""
    input_price_per_1k: float  # $ per 1K input tokens
    output_price_per_1k: float  # $ per 1K output tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost


# Pricing as of 2025
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-32k": ModelPricing(60.00, 120.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-16k": ModelPricing(3.00, 4.00),

    # Anthropic Claude
    "claude-3-opus": ModelPricing(15.00, 75.00),
    "claude-3-sonnet": ModelPricing(3.00, 15.00),
    "claude-3-haiku": ModelPricing(0.25, 1.25),
    "claude-2.1": ModelPricing(8.00, 24.00),
    "claude-2": ModelPricing(8.00, 24.00),
    "claude-instant": ModelPricing(0.80, 2.40),

    # Default fallback
    "default": ModelPricing(1.00, 3.00),
}


@dataclass
class CostReport:
    """Report of costs saved by using mocks"""
    total_saved: float = 0.0
    requests_mocked: int = 0
    input_tokens_mocked: int = 0
    output_tokens_mocked: int = 0
    breakdown_by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    breakdown_by_hour: Dict[str, float] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)

    def track_request(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Track a mocked request and return cost saved"""
        self.requests_mocked += 1
        self.input_tokens_mocked += input_tokens
        self.output_tokens_mocked += output_tokens

        # Get pricing for model
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        cost_saved = pricing.calculate_cost(input_tokens, output_tokens)

        self.total_saved += cost_saved

        # Track by model
        if model not in self.breakdown_by_model:
            self.breakdown_by_model[model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_saved": 0.0
            }

        self.breakdown_by_model[model]["requests"] += 1
        self.breakdown_by_model[model]["input_tokens"] += input_tokens
        self.breakdown_by_model[model]["output_tokens"] += output_tokens
        self.breakdown_by_model[model]["cost_saved"] += cost_saved

        # Track by hour
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        if hour_key not in self.breakdown_by_hour:
            self.breakdown_by_hour[hour_key] = 0.0
        self.breakdown_by_hour[hour_key] += cost_saved

        return cost_saved

    def get_summary(self) -> str:
        """Get a human-readable summary"""
        runtime = (datetime.now() - self.session_start).total_seconds()
        hours = runtime / 3600

        if self.requests_mocked == 0:
            return "No requests mocked yet"

        lines = [
            f"💰 Cost Savings Report",
            f"═══════════════════════",
            f"Total Saved: ${self.total_saved:.2f}",
            f"Requests Mocked: {self.requests_mocked:,}",
            f"Tokens Processed: {self.input_tokens_mocked + self.output_tokens_mocked:,}",
            f"Runtime: {hours:.1f} hours",
            f"",
            f"Savings by Model:",
        ]

        for model, stats in sorted(self.breakdown_by_model.items(),
                                   key=lambda x: x[1]["cost_saved"],
                                   reverse=True):
            lines.append(f"  {model}: ${stats['cost_saved']:.2f} ({stats['requests']} requests)")

        if hours > 0:
            lines.append(f"")
            lines.append(f"Average savings rate: ${self.total_saved / hours:.2f}/hour")

        # Add fun comparisons
        if self.total_saved >= 5:
            lines.append(f"")
            lines.append(f"That's enough for:")
            if self.total_saved >= 5:
                lines.append(f"  ☕ {int(self.total_saved / 5)} cups of coffee")
            if self.total_saved >= 15:
                lines.append(f"  🍕 {int(self.total_saved / 15)} pizzas")
            if self.total_saved >= 50:
                lines.append(f"  🎮 {int(self.total_saved / 50)} indie games")

        return "\n".join(lines)

    def to_json(self) -> Dict:
        """Export report as JSON"""
        return {
            "total_saved": round(self.total_saved, 2),
            "requests_mocked": self.requests_mocked,
            "input_tokens_mocked": self.input_tokens_mocked,
            "output_tokens_mocked": self.output_tokens_mocked,
            "breakdown_by_model": self.breakdown_by_model,
            "breakdown_by_hour": self.breakdown_by_hour,
            "session_start": self.session_start.isoformat(),
            "runtime_hours": (datetime.now() - self.session_start).total_seconds() / 3600
        }


class CostTracker:
    """Global cost tracker for the server"""

    def __init__(self) -> None:
        self.report = CostReport()

    def track(self, model: str, input_tokens: Optional[int] = None,
              output_tokens: Optional[int] = None) -> float:
        """Track a request and return cost saved"""
        # Use defaults if not provided
        if input_tokens is None:
            input_tokens = 100  # Reasonable default
        if output_tokens is None:
            output_tokens = 200  # Reasonable default

        return self.report.track_request(model, input_tokens, output_tokens)

    def reset(self) -> None:
        """Reset tracking"""
        self.report = CostReport()

    def get_report(self) -> CostReport:
        """Get current report"""
        return self.report