from mocktopus import Scenario, OpenAIStubClient, load_yaml

def test_basic_haiku():
    scenario = load_yaml("examples/haiku.yaml")
    client = OpenAIStubClient(scenario)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "please write a haiku"}],
    )
    text = resp.choices[0].message.content
    assert "Eight arms" in text
