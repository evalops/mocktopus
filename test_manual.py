#!/usr/bin/env python3
"""
Manual test script to verify Mocktopus functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from mocktopus import Scenario, Rule
from mocktopus.cost_tracker import CostTracker
from mocktopus.recorder import RecordedInteraction


def test_scenario_matching():
    """Test basic scenario matching"""
    print("Testing scenario matching...")

    scenario = Scenario()
    scenario.add_rule(Rule(
        type="llm.openai",
        when={"messages_contains": "hello"},
        respond={"content": "Hello from test!"}
    ))

    # Test match
    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "hello world"}]
    )
    assert rule is not None, "Should find matching rule"
    assert response["content"] == "Hello from test!"
    print("✓ Basic matching works")

    # Test no match
    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "goodbye"}]
    )
    assert rule is None, "Should not find matching rule"
    print("✓ Non-matching works")


def test_error_scenarios():
    """Test error scenario support"""
    print("\nTesting error scenarios...")

    scenario = Scenario()
    scenario.add_rule(Rule(
        type="llm.openai",
        when={"messages_contains": "error"},
        error={
            "error_type": "rate_limit",
            "message": "Rate limit exceeded",
            "status_code": 429
        }
    ))

    rule, response = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "trigger error"}]
    )
    assert rule is not None, "Should find error rule"
    assert "error_type" in response, "Should return error config"
    assert response["error_type"] == "rate_limit"
    print("✓ Error scenarios work")


def test_cost_tracking():
    """Test cost tracking functionality"""
    print("\nTesting cost tracking...")

    tracker = CostTracker()

    # Track GPT-4 usage
    cost1 = tracker.track("gpt-4", 100, 200)
    assert cost1 > 0, "Should calculate cost"
    print(f"✓ GPT-4 cost tracked: ${cost1:.2f}")

    # Track GPT-3.5 usage
    cost2 = tracker.track("gpt-3.5-turbo", 500, 1000)
    assert cost2 > 0, "Should calculate cost"
    print(f"✓ GPT-3.5 cost tracked: ${cost2:.2f}")

    # Check report
    report = tracker.get_report()
    assert report.requests_mocked == 2
    assert report.total_saved == cost1 + cost2
    print(f"✓ Total saved: ${report.total_saved:.2f}")

    # Get summary
    summary = report.get_summary()
    assert "Cost Savings Report" in summary
    print("\n" + summary)


def test_usage_limits():
    """Test rule usage limits"""
    print("\nTesting usage limits...")

    scenario = Scenario()
    rule = Rule(
        type="llm.openai",
        when={"messages_contains": "limited"},
        respond={"content": "Limited response"},
        times=2
    )
    scenario.add_rule(rule)

    # Should work twice
    for i in range(2):
        matched, _ = scenario.find_llm(
            model="gpt-4",
            messages=[{"role": "user", "content": "limited test"}]
        )
        assert matched is not None, f"Should work on attempt {i+1}"
        matched.consume()
        print(f"✓ Usage {i+1}/2 worked")

    # Should fail on third attempt
    matched, _ = scenario.find_llm(
        model="gpt-4",
        messages=[{"role": "user", "content": "limited test"}]
    )
    assert matched is None, "Should not work after limit"
    print("✓ Usage limit enforced")


def test_recording():
    """Test recording functionality"""
    print("\nTesting recording functionality...")

    interaction = RecordedInteraction(
        timestamp=1234567890,
        request_method="POST",
        request_path="/v1/chat/completions",
        request_headers={"Content-Type": "application/json"},
        request_body={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}]
        },
        response_status=200,
        response_headers={},
        response_body={
            "choices": [{"message": {"content": "Response"}}]
        },
        response_time_ms=250.5,
        model="gpt-4"
    )

    # Test exact match
    assert interaction.matches_request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}]
        },
        fuzzy=False
    )
    print("✓ Exact recording match works")

    # Test fuzzy match
    assert interaction.matches_request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.5  # Extra parameter
        },
        fuzzy=True
    )
    print("✓ Fuzzy recording match works")

    # Test serialization
    data = interaction.to_dict()
    restored = RecordedInteraction.from_dict(data)
    assert restored.model == "gpt-4"
    assert restored.response_time_ms == 250.5
    print("✓ Recording serialization works")


def main():
    """Run all tests"""
    print("=" * 60)
    print("🐙 Mocktopus Manual Test Suite")
    print("=" * 60)

    try:
        test_scenario_matching()
        test_error_scenarios()
        test_cost_tracking()
        test_usage_limits()
        test_recording()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()