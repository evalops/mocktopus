from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import click

from .core import Scenario, load_yaml
from .server import MockServer, ServerMode
from .llm_openai import OpenAIStubClient


@click.group()
def main() -> None:
    """🐙 Mocktopus - Multi-armed mocks for LLM apps"""


@main.command("serve")
@click.option("--scenario", "-s", type=click.Path(exists=True), help="YAML scenario file")
@click.option("--port", "-p", default=8080, show_default=True, help="Server port")
@click.option("--host", default="127.0.0.1", show_default=True, help="Server host")
@click.option("--mode", type=click.Choice(["mock", "record", "replay"]), default="mock", show_default=True,
              help="Server mode")
@click.option("--recordings-dir", type=click.Path(), help="Directory for recordings (record/replay modes)")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for recording mode")
@click.option("--anthropic-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key for recording mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def serve_cmd(scenario: str, port: int, host: str, mode: str, recordings_dir: str,
              openai_key: str, anthropic_key: str, verbose: bool) -> None:
    """
    Start the Mocktopus server to mock LLM APIs.

    Examples:

        # Serve mocks from a scenario file
        mocktopus serve -s examples/haiku.yaml

        # Record real API calls (requires API key)
        mocktopus serve --mode record --recordings-dir ./recordings --openai-key sk-...

        # Replay recorded calls
        mocktopus serve --mode replay --recordings-dir ./recordings
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    server_mode = ServerMode(mode)
    scenario_obj = None

    if server_mode == ServerMode.MOCK and scenario:
        scenario_obj = load_yaml(scenario)
        click.echo(f"📂 Loaded scenario: {scenario} ({len(scenario_obj.rules)} rules)")
    elif server_mode == ServerMode.MOCK:
        click.echo("⚠️  Warning: No scenario file provided, server will return 404 for all requests", err=True)

    server = MockServer(
        scenario=scenario_obj,
        mode=server_mode,
        recordings_dir=Path(recordings_dir) if recordings_dir else None,
        real_openai_key=openai_key,
        real_anthropic_key=anthropic_key,
        host=host,
        port=port
    )

    click.echo(f"🐙 Starting Mocktopus server on http://{host}:{port}")
    click.echo(f"📝 Mode: {mode}")
    click.echo(f"🔌 OpenAI endpoint: http://{host}:{port}/v1/chat/completions")
    click.echo(f"🔌 Anthropic endpoint: http://{host}:{port}/v1/messages")
    click.echo("\nPress Ctrl+C to stop the server")

    try:
        server.run()
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")


@main.command("simulate")
@click.option("--scenario", "-s", "scenario_file", required=True, type=click.Path(exists=True),
              help="YAML scenario file")
@click.option("--model", default="gpt-4o-mini", show_default=True, help="Model to simulate")
@click.option("--prompt", required=True, help="User prompt to simulate")
@click.option("--stream/--no-stream", default=False, show_default=True, help="Stream the response")
def simulate_cmd(scenario_file: str, model: str, prompt: str, stream: bool) -> None:
    """
    Simulate an LLM call using a scenario file (without starting a server).

    Example:
        mocktopus simulate -s examples/haiku.yaml --prompt "write a haiku"
    """
    scenario = load_yaml(scenario_file)
    client = OpenAIStubClient(scenario)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

    try:
        out = client.chat.completions.create(model=model, messages=messages, stream=stream)

        if stream:
            for ev in out:
                delta = (ev.choices[0].delta.content or "")
                sys.stdout.write(delta)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            click.echo(out.choices[0].message.content)
    except KeyError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@main.command("validate")
@click.argument("scenario_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed validation information")
@click.option("--schema-only", is_flag=True, help="Only validate schema, skip rule logic checks")
def validate_cmd(scenario_file: str, verbose: bool, schema_only: bool) -> None:
    """
    Validate a scenario YAML file with comprehensive schema and logic checks.

    This command validates both the YAML structure and the logical consistency
    of your Mocktopus scenario file.

    Examples:
        mocktopus validate examples/haiku.yaml
        mocktopus validate config.yaml --verbose
        mocktopus validate config.yaml --schema-only
    """
    from .core import validate_scenario_data
    import yaml

    click.echo(f"🔍 Validating scenario file: {scenario_file}")

    # First, check if file is valid YAML
    try:
        with open(scenario_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        click.echo("✅ Valid YAML syntax")
    except yaml.YAMLError as e:
        click.echo(f"❌ YAML syntax error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error reading file: {e}", err=True)
        sys.exit(1)

    # Schema validation
    click.echo("🔬 Checking schema compliance...")
    validation_errors = validate_scenario_data(data)

    if validation_errors:
        click.echo(f"❌ Schema validation failed ({len(validation_errors)} errors):", err=True)
        for i, error in enumerate(validation_errors, 1):
            click.echo(f"  {i}. {error}", err=True)

        click.echo(f"\n💡 Schema validation tips:", err=True)
        click.echo("   • Each rule must have 'type' and 'when' fields", err=True)
        click.echo("   • Rules must have either 'respond' or 'error' field", err=True)
        click.echo("   • Version must be 1", err=True)
        click.echo("   • Use 'mocktopus init' to see valid examples", err=True)
        sys.exit(1)

    click.echo("✅ Schema validation passed")

    if schema_only:
        click.echo("🎉 Schema-only validation completed successfully!")
        return

    # Load scenario for additional validation
    try:
        scenario = load_yaml(scenario_file)
        click.echo(f"✅ Successfully loaded scenario with {len(scenario.rules)} rules")
    except Exception as e:
        click.echo(f"❌ Error loading scenario: {e}", err=True)
        sys.exit(1)

    # Logical validation
    click.echo("🧠 Performing logical validation...")
    warnings = []
    issues = []

    # Check for empty rules
    if len(scenario.rules) == 0:
        warnings.append("No rules defined - server will return 404 for all requests")

    # Check for rule conflicts and patterns
    broad_patterns = []
    specific_patterns = []

    for i, rule in enumerate(scenario.rules):
        rule_num = i + 1

        # Check if rule has required fields in when clause
        if not rule.when:
            issues.append(f"Rule {rule_num}: Empty 'when' clause")
            continue

        # Categorize rules by specificity
        if rule.when.get('model') == "*":
            broad_patterns.append(rule_num)
        else:
            specific_patterns.append(rule_num)

        # Check for regex validity
        if rule.when.get('messages_regex'):
            try:
                import re
                re.compile(rule.when['messages_regex'])
            except re.error as e:
                issues.append(f"Rule {rule_num}: Invalid regex pattern '{rule.when['messages_regex']}': {e}")

        # Check response completeness
        if hasattr(rule, 'respond') and rule.respond:
            has_content = rule.respond.get('content') is not None
            has_tool_calls = rule.respond.get('tool_calls') is not None
            has_embeddings = rule.respond.get('embeddings') is not None
            has_images = rule.respond.get('images') is not None

            if not any([has_content, has_tool_calls, has_embeddings, has_images]):
                warnings.append(f"Rule {rule_num}: Response has no content, tool_calls, embeddings, or images")

        # Check usage tokens
        if hasattr(rule, 'respond') and rule.respond and rule.respond.get('usage'):
            usage = rule.respond['usage']
            if isinstance(usage, dict):
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                if input_tokens == 0 and output_tokens == 0:
                    warnings.append(f"Rule {rule_num}: Zero token usage - won't contribute to cost tracking")

    # Check rule ordering
    if broad_patterns and specific_patterns:
        first_broad = min(broad_patterns) if broad_patterns else float('inf')
        last_specific = max(specific_patterns) if specific_patterns else 0
        if first_broad < last_specific:
            warnings.append(f"Broad pattern rule (rule {first_broad}) appears before specific rules - may mask later rules")

    # Check for fallback rule
    has_fallback = any(
        rule.when.get('model') == "*"
        for rule in scenario.rules
    )
    if not has_fallback:
        warnings.append("No fallback rule (model: '*') found - unmatched requests will return 404")

    # Report results
    if issues:
        click.echo(f"❌ Found {len(issues)} critical issues:", err=True)
        for issue in issues:
            click.echo(f"  • {issue}", err=True)

    if warnings:
        click.echo(f"⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            click.echo(f"  • {warning}")

    # Detailed rule breakdown if verbose
    if verbose:
        click.echo(f"\n📋 Rule Details:")
        for i, rule in enumerate(scenario.rules, 1):
            click.echo(f"\n  Rule {i}: {rule.type}")
            click.echo(f"    When: {dict(rule.when)}")
            if hasattr(rule, 'respond') and rule.respond:
                click.echo(f"    Response: {len(str(rule.respond.get('content', '')))} chars")
            if hasattr(rule, 'error') and rule.error:
                click.echo(f"    Error: {rule.error.get('error_type', 'unknown')}")
            if rule.times:
                click.echo(f"    Times: {rule.times}")

    # Final summary
    click.echo(f"\n📊 Validation Summary:")
    click.echo(f"  Total rules: {len(scenario.rules)}")
    click.echo(f"  LLM rules: {len([r for r in scenario.rules if r.type.startswith('llm.')])}")
    click.echo(f"  Critical issues: {len(issues)}")
    click.echo(f"  Warnings: {len(warnings)}")

    if issues:
        click.echo(f"\n❌ Validation failed due to critical issues")
        sys.exit(1)
    elif warnings:
        click.echo(f"\n⚠️  Validation completed with warnings")
    else:
        click.echo(f"\n🎉 Validation passed - scenario is ready to use!")

    # Suggest next steps
    if not issues:
        click.echo(f"\n🚀 Next steps:")
        click.echo(f"  • Test: mocktopus simulate -s {scenario_file} --prompt 'hello'")
        click.echo(f"  • Serve: mocktopus serve -s {scenario_file}")
        click.echo(f"  • Debug: mocktopus explain -s {scenario_file} --prompt 'your test prompt'")


@main.command("init")
@click.option("--template", type=click.Choice(["basic", "rag", "agents", "multimodal", "enterprise"]),
              default="basic", help="Project template to initialize")
@click.option("--output", "-o", default="mocktopus.yaml", help="Output scenario file name")
@click.option("--force", is_flag=True, help="Overwrite existing files")
def init_cmd(template: str, output: str, force: bool) -> None:
    """
    Initialize a new Mocktopus project with starter scenarios.

    Templates:
      basic      - Simple chat completion mocking
      rag        - RAG/embeddings application testing
      agents     - Multi-step agent workflow testing
      multimodal - Image, audio, and vision API testing
      enterprise - Advanced features with error handling

    Examples:
        mocktopus init                           # Basic template
        mocktopus init --template rag -o rag.yaml
        mocktopus init --template enterprise --force
    """

    if not force and Path(output).exists():
        click.echo(f"❌ File {output} already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    templates = {
        "basic": _generate_basic_template(),
        "rag": _generate_rag_template(),
        "agents": _generate_agents_template(),
        "multimodal": _generate_multimodal_template(),
        "enterprise": _generate_enterprise_template()
    }

    content = templates[template]

    try:
        with open(output, 'w') as f:
            f.write(content)

        click.echo(f"✅ Created {output} with {template} template")
        click.echo(f"🚀 Start server: mocktopus serve -s {output}")
        click.echo(f"🧪 Test scenario: mocktopus simulate -s {output} --prompt 'hello'")

    except Exception as e:
        click.echo(f"❌ Error creating {output}: {e}", err=True)
        sys.exit(1)


@main.command("doctor")
@click.option("--scenario", "-s", type=click.Path(), help="YAML scenario file to diagnose")
@click.option("--port", "-p", default=8080, help="Port to check availability")
@click.option("--fix", is_flag=True, help="Attempt to fix common issues")
def doctor_cmd(scenario: Optional[str], port: int, fix: bool) -> None:
    """
    Diagnose Mocktopus configuration and environment issues.

    Examples:
        mocktopus doctor                    # Check general environment
        mocktopus doctor -s config.yaml    # Diagnose specific scenario
        mocktopus doctor --fix             # Try to fix issues automatically
    """

    click.echo("🩺 Running Mocktopus diagnostics...\n")

    issues_found = 0
    warnings = 0

    # Check Python environment
    click.echo("📦 Checking Python environment...")
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"  ✅ Python {python_version}")

    # Check required dependencies
    dependencies = ["aiohttp", "click", "PyYAML"]
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_").lower())
            click.echo(f"  ✅ {dep} installed")
        except ImportError:
            click.echo(f"  ❌ {dep} missing", err=True)
            issues_found += 1
            if fix:
                click.echo(f"    💡 Run: pip install {dep}")

    # Check port availability
    click.echo(f"\n🔌 Checking port {port} availability...")
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('127.0.0.1', port))
        if result == 0:
            click.echo(f"  ⚠️  Port {port} is in use", err=True)
            warnings += 1
            # Try to find available port
            for test_port in range(port + 1, port + 100):
                result = s.connect_ex(('127.0.0.1', test_port))
                if result != 0:
                    click.echo(f"    💡 Port {test_port} is available")
                    break
        else:
            click.echo(f"  ✅ Port {port} available")

    # Check scenario file if provided
    if scenario:
        click.echo(f"\n📄 Analyzing scenario file: {scenario}")
        scenario_path = Path(scenario)

        if not scenario_path.exists():
            click.echo(f"  ❌ File does not exist: {scenario}", err=True)
            issues_found += 1
            if fix and scenario == "mocktopus.yaml":
                click.echo("    💡 Run: mocktopus init")
        else:
            # Check file permissions
            if not scenario_path.is_file():
                click.echo(f"  ❌ Not a file: {scenario}", err=True)
                issues_found += 1
            elif not scenario_path.stat().st_size:
                click.echo(f"  ❌ File is empty: {scenario}", err=True)
                issues_found += 1
            else:
                click.echo(f"  ✅ File exists and readable")

                # Validate YAML syntax
                try:
                    scenario_obj = load_yaml(str(scenario_path))
                    click.echo(f"  ✅ Valid YAML syntax")

                    # Analyze rules
                    rule_count = len(scenario_obj.rules)
                    click.echo(f"  📊 Found {rule_count} rules")

                    if rule_count == 0:
                        click.echo("  ⚠️  No rules defined", err=True)
                        warnings += 1

                    # Check for common issues
                    llm_rules = [r for r in scenario_obj.rules if r.type.startswith("llm.")]
                    if not llm_rules:
                        click.echo("  ⚠️  No LLM rules found", err=True)
                        warnings += 1

                    # Check for rule conflicts
                    broad_patterns = []
                    specific_patterns = []

                    for i, rule in enumerate(scenario_obj.rules):
                        if hasattr(rule.when, 'model') and rule.when.model == "*":
                            broad_patterns.append(i + 1)
                        else:
                            specific_patterns.append(i + 1)

                    if broad_patterns and specific_patterns:
                        if min(broad_patterns) < max(specific_patterns):
                            click.echo(f"  ⚠️  Broad pattern rule (rule {min(broad_patterns)}) may override specific rules", err=True)
                            warnings += 1
                            if fix:
                                click.echo("    💡 Move broad patterns (model: '*') to the end of rules list")

                    # Check for unused patterns
                    has_fallback = any(
                        hasattr(rule.when, 'model') and rule.when.model == "*"
                        for rule in scenario_obj.rules
                    )
                    if not has_fallback:
                        click.echo("  ⚠️  No fallback rule (model: '*') found", err=True)
                        warnings += 1
                        if fix:
                            click.echo("    💡 Add a fallback rule to handle unmatched requests")

                    # Check response completeness
                    for i, rule in enumerate(scenario_obj.rules):
                        if hasattr(rule, 'respond') and rule.respond:
                            if not hasattr(rule.respond, 'content') or not rule.respond.content:
                                if not hasattr(rule.respond, 'tool_calls') or not rule.respond.tool_calls:
                                    click.echo(f"  ⚠️  Rule {i+1} has empty response content", err=True)
                                    warnings += 1

                except Exception as e:
                    click.echo(f"  ❌ Invalid scenario file: {e}", err=True)
                    issues_found += 1

    # Check for common configuration files
    click.echo(f"\n📁 Checking for configuration files...")
    config_files = ["mocktopus.yaml", "scenarios/", "examples/"]
    found_configs = []

    for config in config_files:
        if Path(config).exists():
            found_configs.append(config)
            click.echo(f"  ✅ Found {config}")

    if not found_configs:
        click.echo("  ⚠️  No configuration files found", err=True)
        warnings += 1
        if fix:
            click.echo("    💡 Run: mocktopus init")

    # Network connectivity test (basic)
    click.echo(f"\n🌐 Testing network connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen('https://httpbin.org/status/200', timeout=5)
        click.echo("  ✅ Internet connectivity OK")
    except Exception:
        click.echo("  ⚠️  Limited network connectivity", err=True)
        warnings += 1
        click.echo("    💡 This may affect recording mode functionality")

    # Memory and performance check (basic)
    click.echo(f"\n💾 System resources...")
    try:
        # Use a lighter approach without requiring psutil
        import resource
        memory_limit = resource.getrlimit(resource.RLIMIT_AS)[0]
        if memory_limit != resource.RLIM_INFINITY:
            click.echo(f"  ✅ Memory limit configured: {memory_limit // (1024*1024)}MB")
        else:
            click.echo("  ✅ No memory limits detected")
    except Exception:
        click.echo("  ✅ System resources check passed")

    # Summary
    click.echo(f"\n📋 Diagnostic Summary")
    click.echo("=" * 30)

    if issues_found == 0 and warnings == 0:
        click.echo("🎉 All checks passed! Your Mocktopus setup looks great.")
    else:
        if issues_found > 0:
            click.echo(f"❌ {issues_found} critical issue(s) found")
        if warnings > 0:
            click.echo(f"⚠️  {warnings} warning(s) found")

        click.echo(f"\n💡 Quick fixes:")
        if issues_found > 0:
            click.echo(f"  - Install missing dependencies")
            if scenario and not Path(scenario).exists():
                click.echo(f"  - Create scenario file: mocktopus init -o {scenario}")

        if warnings > 0 and not found_configs:
            click.echo(f"  - Initialize project: mocktopus init")

        if fix and (issues_found > 0 or warnings > 0):
            click.echo(f"\n🔧 Some issues can be auto-fixed. Re-run with --fix for suggestions.")

    if issues_found > 0:
        sys.exit(1)


@main.command("explain")
@click.option("--scenario", "-s", required=True, type=click.Path(exists=True), help="YAML scenario file")
@click.option("--model", default="gpt-4o-mini", help="Model to test against")
@click.option("--prompt", required=True, help="Test prompt to match against rules")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed matching process")
def explain_cmd(scenario: str, model: str, prompt: str, verbose: bool) -> None:
    """
    Explain which rule would match for a given request and why.

    This is invaluable for debugging rule matching issues and understanding
    how Mocktopus processes requests.

    Examples:
        mocktopus explain -s config.yaml --prompt "hello world"
        mocktopus explain -s config.yaml --model gpt-4 --prompt "help me" -v
    """

    click.echo(f"🔍 Analyzing rule matching for scenario: {scenario}\n")

    try:
        scenario_obj = load_yaml(scenario)
        messages = [{"role": "user", "content": prompt}]

        click.echo(f"📝 Request Details:")
        click.echo(f"  Model: {model}")
        click.echo(f"  Prompt: \"{prompt}\"")
        click.echo(f"  Messages: {messages}")
        click.echo()

        click.echo(f"📊 Scenario Analysis:")
        click.echo(f"  Total rules: {len(scenario_obj.rules)}")
        llm_rules = [r for r in scenario_obj.rules if r.type.startswith("llm.")]
        click.echo(f"  LLM rules: {len(llm_rules)}")
        click.echo()

        # Test rule matching step by step
        click.echo(f"🎯 Rule Matching Process:")
        click.echo("=" * 40)

        matched_rule = None
        match_index = None

        for i, rule in enumerate(scenario_obj.rules):
            rule_num = i + 1
            click.echo(f"\n📋 Rule {rule_num}: {rule.type}")

            if verbose:
                click.echo(f"    Times available: {rule.times if hasattr(rule, 'times') else 'unlimited'}")

            # Check rule type
            if not rule.type.startswith("llm."):
                click.echo(f"    ⏭️  Skipping (not an LLM rule)")
                continue

            # Test matching criteria
            matches = True
            match_reasons = []
            no_match_reasons = []

            # Check model matching
            if hasattr(rule.when, 'model') and rule.when.model:
                model_pattern = rule.when.model
                if model_pattern == "*":
                    match_reasons.append(f"Model matches wildcard '*'")
                elif model_pattern.endswith("*"):
                    prefix = model_pattern[:-1]
                    if model.startswith(prefix):
                        match_reasons.append(f"Model '{model}' matches pattern '{model_pattern}'")
                    else:
                        matches = False
                        no_match_reasons.append(f"Model '{model}' doesn't match pattern '{model_pattern}'")
                elif model_pattern == model:
                    match_reasons.append(f"Model matches exactly: '{model}'")
                else:
                    matches = False
                    no_match_reasons.append(f"Model '{model}' doesn't match required '{model_pattern}'")
            else:
                match_reasons.append("No model constraint")

            # Check message content matching
            if hasattr(rule.when, 'messages_contains') and rule.when.messages_contains:
                search_term = rule.when.messages_contains.lower()
                message_content = ' '.join([msg.get('content', '') for msg in messages]).lower()
                if search_term in message_content:
                    match_reasons.append(f"Message contains '{rule.when.messages_contains}'")
                else:
                    matches = False
                    no_match_reasons.append(f"Message doesn't contain '{rule.when.messages_contains}'")

            # Check regex matching
            if hasattr(rule.when, 'messages_regex') and rule.when.messages_regex:
                import re
                message_content = ' '.join([msg.get('content', '') for msg in messages])
                try:
                    if re.search(rule.when.messages_regex, message_content, re.IGNORECASE):
                        match_reasons.append(f"Message matches regex: /{rule.when.messages_regex}/i")
                    else:
                        matches = False
                        no_match_reasons.append(f"Message doesn't match regex: /{rule.when.messages_regex}/i")
                except re.error as e:
                    matches = False
                    no_match_reasons.append(f"Invalid regex pattern: {e}")

            # Check endpoint matching (future feature)
            if hasattr(rule.when, 'endpoint') and rule.when.endpoint:
                # This would be checked against actual endpoint in real scenario
                match_reasons.append(f"Endpoint pattern: {rule.when.endpoint} (simulation)")

            # Display results
            if matches:
                click.echo(f"    ✅ MATCH!")
                if verbose and match_reasons:
                    for reason in match_reasons:
                        click.echo(f"       • {reason}")

                if not matched_rule:  # First match wins
                    matched_rule = rule
                    match_index = rule_num
                    click.echo(f"    🏆 This rule will be used!")
                else:
                    click.echo(f"    ⚠️  Rule matches but won't be used (rule {match_index} already matched)")
            else:
                click.echo(f"    ❌ No match")
                if verbose and no_match_reasons:
                    for reason in no_match_reasons:
                        click.echo(f"       • {reason}")

        # Show final result
        click.echo(f"\n🎉 Final Result:")
        click.echo("=" * 40)

        if matched_rule:
            click.echo(f"✅ Rule {match_index} will be used")
            click.echo(f"   Type: {matched_rule.type}")

            if hasattr(matched_rule, 'respond') and matched_rule.respond:
                respond = matched_rule.respond
                if hasattr(respond, 'content') and respond.content:
                    content_preview = respond.content[:100] + "..." if len(respond.content) > 100 else respond.content
                    click.echo(f"   Response: \"{content_preview}\"")

                if hasattr(respond, 'tool_calls') and respond.tool_calls:
                    click.echo(f"   Tool calls: {len(respond.tool_calls)} function(s)")
                    if verbose:
                        for i, tool_call in enumerate(respond.tool_calls):
                            if isinstance(tool_call, dict):
                                func_name = tool_call.get('function', {}).get('name', 'unknown')
                                click.echo(f"     {i+1}. {func_name}()")

                if hasattr(respond, 'usage') and respond.usage:
                    usage = respond.usage
                    input_tokens = getattr(usage, 'input_tokens', 0) if hasattr(usage, 'input_tokens') else usage.get('input_tokens', 0)
                    output_tokens = getattr(usage, 'output_tokens', 0) if hasattr(usage, 'output_tokens') else usage.get('output_tokens', 0)
                    click.echo(f"   Usage: {input_tokens} input + {output_tokens} output tokens")

            if hasattr(matched_rule, 'error') and matched_rule.error:
                click.echo(f"   ⚠️  This rule will return an error")
                error = matched_rule.error
                error_type = getattr(error, 'error_type', 'unknown') if hasattr(error, 'error_type') else error.get('error_type', 'unknown')
                message = getattr(error, 'message', 'No message') if hasattr(error, 'message') else error.get('message', 'No message')
                click.echo(f"       Type: {error_type}")
                click.echo(f"       Message: {message}")

        else:
            click.echo(f"❌ No rules match this request")
            click.echo(f"   The server would return a 404 or generic error")
            click.echo(f"\n💡 Suggestions:")
            click.echo(f"   • Add a fallback rule with model: '*'")
            click.echo(f"   • Check your model patterns and message matching")
            click.echo(f"   • Use 'mocktopus doctor -s {scenario}' to check for issues")

        # Show potential improvements
        if verbose and matched_rule:
            click.echo(f"\n🔧 Optimization Suggestions:")

            # Check if there are more specific rules after this one
            remaining_rules = scenario_obj.rules[match_index:]
            specific_matches = []
            for rule in remaining_rules:
                if (hasattr(rule.when, 'messages_contains') and rule.when.messages_contains and
                    rule.when.messages_contains.lower() in prompt.lower()):
                    specific_matches.append(rule)

            if specific_matches:
                click.echo(f"   ⚠️  Consider moving more specific rules before broad patterns")

            # Check for unused rules
            total_rules = len(scenario_obj.rules)
            if match_index and match_index < total_rules:
                unused_count = total_rules - match_index
                click.echo(f"   📊 {unused_count} rule(s) after the match won't be evaluated")

    except Exception as e:
        click.echo(f"❌ Error analyzing scenario: {e}", err=True)
        sys.exit(1)


@main.command("example")
@click.option("--type", "example_type", type=click.Choice(["basic", "streaming", "tools", "multi-model"]),
              default="basic", help="Type of example to generate")
def example_cmd(example_type: str) -> None:
    """
    Generate example scenario files.

    Example:
        mocktopus example --type basic > my-scenario.yaml
    """
    examples = {
        "basic": """version: 1
meta:
  description: Basic chat completion mock

rules:
  - type: llm.openai
    when:
      model: "gpt-*"
      messages_contains: "hello"
    respond:
      content: "Hello! How can I help you today?"
      usage:
        input_tokens: 10
        output_tokens: 8
""",
        "streaming": """version: 1
meta:
  description: Streaming response example

rules:
  - type: llm.openai
    when:
      model: "*"
      messages_regex: "stream|realtime"
    respond:
      content: "This is a streaming response that will be sent chunk by chunk."
      delay_ms: 50
      chunk_size: 5
""",
        "tools": """version: 1
meta:
  description: Function calling example

rules:
  - type: llm.openai
    when:
      model: "gpt-4*"
      messages_contains: "weather"
    respond:
      content: null
      tool_calls:
        - id: "call_abc123"
          type: "function"
          function:
            name: "get_weather"
            arguments: '{"location": "San Francisco"}'
""",
        "multi_model": """version: 1
meta:
  description: Multi-model routing example

rules:
  # GPT-4 response
  - type: llm.openai
    when:
      model: "gpt-4*"
      messages_contains: "explain"
    respond:
      content: "Here's a detailed explanation..."

  # GPT-3.5 response
  - type: llm.openai
    when:
      model: "gpt-3.5*"
      messages_contains: "explain"
    respond:
      content: "Here's a simple explanation..."

  # Claude response
  - type: llm.openai
    when:
      model: "claude-*"
      messages_contains: "explain"
    respond:
      content: "I'll explain this step by step..."

  # Default fallback
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "I understand your request."
"""
    }

    click.echo(examples.get(example_type, examples["basic"]))


# Template generation functions for init command

def _generate_basic_template() -> str:
    """Generate basic starter template"""
    return """version: 1
meta:
  description: Basic Mocktopus scenario for getting started
  created: "Generated by mocktopus init"

rules:
  # Welcome message
  - type: llm.openai
    when:
      messages_contains: "hello"
    respond:
      content: "Hello! I'm a mocked LLM response. This is working great!"
      usage:
        input_tokens: 10
        output_tokens: 12

  # Help request
  - type: llm.openai
    when:
      messages_regex: "help|what can you do"
    respond:
      content: |
        I'm a mock LLM that can help you test your application! Try asking me about:
        - Weather information
        - Simple calculations
        - General questions
      usage:
        input_tokens: 15
        output_tokens: 25

  # Default fallback
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "I understand your request, but I'm just a mock response. Update this scenario to customize my behavior!"
      usage:
        input_tokens: 20
        output_tokens: 22
"""

def _generate_rag_template() -> str:
    """Generate RAG/embeddings template"""
    return """version: 1
meta:
  description: RAG application testing with embeddings and retrieval
  use_cases: ["document_qa", "semantic_search", "knowledge_retrieval"]

rules:
  # Document embedding requests
  - type: llm.openai
    when:
      # Will match embeddings endpoint when implemented
      endpoint: "/v1/embeddings"
    respond:
      embeddings:
        - embedding: [0.1, 0.2, -0.3, 0.4]  # Mock 4-dimensional embedding
      usage:
        input_tokens: 50
        output_tokens: 0

  # Document Q&A with context
  - type: llm.openai
    when:
      messages_regex: "according to|based on|document says"
    respond:
      content: |
        Based on the retrieved documents, here's what I found:

        **Key Points:**
        - Document section 2.3 mentions the relevant information
        - The data shows a 23% increase in efficiency
        - Implementation requires 2-3 weeks

        **Sources:** doc_1.pdf (page 15), doc_2.pdf (page 8)
      usage:
        input_tokens: 120
        output_tokens: 45

  # Semantic search queries
  - type: llm.openai
    when:
      messages_contains: "find documents about"
    respond:
      content: |
        I found 3 relevant documents:
        1. "Introduction to Machine Learning" (similarity: 0.89)
        2. "Deep Learning Fundamentals" (similarity: 0.76)
        3. "Neural Networks Overview" (similarity: 0.71)

        Would you like me to retrieve specific information from any of these?
      usage:
        input_tokens: 25
        output_tokens: 35

  # No context available fallback
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "I don't have enough context in my knowledge base to answer that question. Please provide more specific documents or context."
      usage:
        input_tokens: 30
        output_tokens: 25
"""

def _generate_agents_template() -> str:
    """Generate multi-step agent workflow template"""
    return """version: 1
meta:
  description: Multi-step agent testing with tool calls and workflows
  use_cases: ["research_agent", "task_planning", "tool_orchestration"]

rules:
  # Research planning
  - type: llm.openai
    when:
      messages_regex: "research|investigate|find information about"
    respond:
      content: "I'll help you research this topic. Let me break this down into steps and gather information."
      tool_calls:
        - id: "call_plan_001"
          type: "function"
          function:
            name: "create_research_plan"
            arguments: '{"topic": "user query", "steps": ["web_search", "summarize", "fact_check"]}'
      usage:
        input_tokens: 35
        output_tokens: 25

  # Web search tool call
  - type: llm.openai
    when:
      messages_contains: "search the web"
    respond:
      content: null
      tool_calls:
        - id: "call_search_002"
          type: "function"
          function:
            name: "web_search"
            arguments: '{"query": "latest information", "max_results": 5, "date_range": "past_week"}'
      usage:
        input_tokens: 20
        output_tokens: 30

  # Data analysis tool call
  - type: llm.openai
    when:
      messages_regex: "analyze|process data"
    respond:
      content: null
      tool_calls:
        - id: "call_analyze_003"
          type: "function"
          function:
            name: "analyze_data"
            arguments: '{"data_source": "search_results", "analysis_type": "summary_statistics"}'
      usage:
        input_tokens: 25
        output_tokens: 28

  # Multi-step workflow
  - type: llm.openai
    when:
      messages_contains: "complete task"
    respond:
      content: "I'll complete this task step by step using multiple tools."
      tool_calls:
        - id: "call_multi_001"
          type: "function"
          function:
            name: "task_planner"
            arguments: '{"task": "user_request", "priority": "high"}'
        - id: "call_multi_002"
          type: "function"
          function:
            name: "resource_allocator"
            arguments: '{"resources": ["web_search", "data_analysis", "report_generation"]}'
      usage:
        input_tokens: 40
        output_tokens: 45

  # Default agent response
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "I'm an agent mock ready to help! I can research topics, analyze data, and orchestrate multi-step workflows. What task would you like me to help with?"
      usage:
        input_tokens: 30
        output_tokens: 35
"""

def _generate_multimodal_template() -> str:
    """Generate multimodal (images, audio) template"""
    return """version: 1
meta:
  description: Multimodal application testing (images, audio, vision)
  use_cases: ["image_generation", "vision_analysis", "audio_processing"]

rules:
  # Image generation requests
  - type: llm.openai
    when:
      # Will match image generation endpoint when implemented
      endpoint: "/v1/images/generations"
    respond:
      images:
        - url: "https://example.com/generated-image-001.png"
          revised_prompt: "A photorealistic image of the requested subject with enhanced lighting"
      usage:
        input_tokens: 25
        output_tokens: 0

  # Vision/image analysis
  - type: llm.openai
    when:
      messages_regex: "analyze image|what.* see|describe.*picture"
    respond:
      content: |
        Looking at this image, I can see:

        **Objects detected:**
        - Person (confidence: 0.95)
        - Car (confidence: 0.87)
        - Building (confidence: 0.76)

        **Scene description:**
        This appears to be an urban street scene during daytime. The lighting suggests it's mid-afternoon with clear weather conditions.

        **Technical details:**
        - Resolution: 1024x768
        - Format: JPEG
        - Quality score: 8.5/10
      usage:
        input_tokens: 45
        output_tokens: 65

  # Audio processing
  - type: llm.openai
    when:
      messages_regex: "transcribe|speech.*text|audio"
    respond:
      content: |
        **Audio Transcription:**
        "Hello, this is a transcribed audio message. The speech was clear with minimal background noise."

        **Audio Analysis:**
        - Duration: 12.3 seconds
        - Language: English (confidence: 0.98)
        - Speaker count: 1
        - Background noise level: Low
      usage:
        input_tokens: 30
        output_tokens: 40

  # Text-to-speech
  - type: llm.openai
    when:
      messages_contains: "text to speech"
    respond:
      content: "I've generated the audio file for your text. In a real scenario, this would return an audio file URL."
      audio_url: "https://example.com/generated-speech.mp3"
      usage:
        input_tokens: 20
        output_tokens: 15

  # Default multimodal response
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "I'm a multimodal mock that can handle images, audio, and vision tasks! Try asking me to generate images, analyze pictures, or process audio."
      usage:
        input_tokens: 25
        output_tokens: 30
"""

def _generate_enterprise_template() -> str:
    """Generate enterprise-grade template with advanced features"""
    return """version: 1
meta:
  description: Enterprise-grade Mocktopus scenario with advanced error handling
  environment: "production_testing"
  author: "DevOps Team"

rules:
  # Rate limiting simulation
  - type: llm.openai
    when:
      messages_contains: "rate limit test"
    times: 3  # Allow 3 requests then fail
    respond:
      content: "This is request within rate limit"
      usage:
        input_tokens: 10
        output_tokens: 8

  # After rate limit exceeded
  - type: llm.openai
    when:
      messages_contains: "rate limit test"
    error:
      error_type: "rate_limit"
      message: "Rate limit exceeded: 100 requests per minute. Retry after 60 seconds."
      status_code: 429
      retry_after: 60
      delay_ms: 500

  # Authentication failure
  - type: llm.openai
    when:
      messages_contains: "auth failure"
    error:
      error_type: "authentication"
      message: "Invalid API key. Please check your credentials."
      status_code: 401
      delay_ms: 100

  # Timeout simulation for load testing
  - type: llm.openai
    when:
      messages_contains: "timeout test"
    error:
      error_type: "timeout"
      message: "Request timeout after 30 seconds"
      status_code: 504
      delay_ms: 5000

  # Model-specific responses
  - type: llm.openai
    when:
      model: "gpt-4*"
      messages_regex: "complex|detailed|analysis"
    respond:
      content: |
        **Detailed Analysis (GPT-4 Level)**

        I'll provide a comprehensive analysis with multiple perspectives:

        1. **Technical Assessment**: High complexity requiring advanced reasoning
        2. **Risk Analysis**: Medium risk with mitigation strategies available
        3. **Recommendations**: Implement gradual rollout with monitoring
        4. **Timeline**: 2-3 weeks for full implementation

        This response demonstrates GPT-4 level detail and reasoning.
      usage:
        input_tokens: 50
        output_tokens: 85

  # Cost optimization - cheaper model responses
  - type: llm.openai
    when:
      model: "gpt-3.5*"
    respond:
      content: "Quick response from cost-optimized model. This saves ~90% on API costs during testing."
      usage:
        input_tokens: 25
        output_tokens: 18

  # Content filtering
  - type: llm.openai
    when:
      messages_regex: "inappropriate|filtered content"
    error:
      error_type: "content_filter"
      message: "Content filtered by safety system"
      code: "content_policy_violation"
      status_code: 422

  # Fallback with monitoring
  - type: llm.openai
    when:
      model: "*"
    respond:
      content: "Enterprise mock response with full monitoring and error handling capabilities. Ready for production testing scenarios."
      usage:
        input_tokens: 30
        output_tokens: 25
      metadata:
        environment: "staging"
        version: "1.0"
        monitored: true
"""


if __name__ == "__main__":
    main()