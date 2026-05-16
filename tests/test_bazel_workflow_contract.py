from __future__ import annotations

from pathlib import Path


def test_bazel_rbe_workflow_installs_uv_before_bazel_checks() -> None:
    workflow = Path(".github/workflows/bazel-rbe.yml").read_text()

    uv_setup = workflow.index("astral-sh/setup-uv")
    bazel_check = workflow.index("make bazel-check")

    assert uv_setup < bazel_check
