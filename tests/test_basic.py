import yaml

from mocktopus import OpenAIStubClient, load_yaml
from mocktopus.core import validate_scenario_data


def test_basic_haiku() -> None:
    scenario = load_yaml("examples/haiku.yaml")
    client = OpenAIStubClient(scenario)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "please write a haiku"}],
    )
    text = resp.choices[0].message.content
    assert "Eight arms" in text


def test_error_scenarios_round_trip_to_yaml() -> None:
    scenario = load_yaml("examples/errors.yaml")

    dumped = yaml.safe_load(scenario.to_yaml())

    assert dumped["rules"][0]["error"]["error_type"] == "rate_limit"
    assert "respond" not in dumped["rules"][0]
    assert validate_scenario_data(dumped) == []


def test_scenario_validation_rejects_bad_error_config() -> None:
    errors = validate_scenario_data(
        {
            "version": 1,
            "rules": [
                {
                    "type": "llm.openai",
                    "when": {"messages_contains": "bad"},
                    "error": {
                        "error_type": "explosion",
                        "message": "bad",
                        "status_code": 200,
                        "delay_ms": -1,
                    },
                }
            ],
        }
    )

    assert any("error_type" in error for error in errors)
    assert any("status_code" in error for error in errors)
    assert any("delay_ms" in error for error in errors)
