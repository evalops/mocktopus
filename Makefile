
.PHONY: fmt test requirements-lock bazel-format bazel-mod-tidy bazel-check bazel-test bazel-test-remote bazel-rbe-smoke

fmt:
	ruff check --fix || true

test:
	pytest -q

requirements-lock:
	uv pip compile pyproject.toml --extra test -o requirements_lock.txt

bazel-format:
	buildifier BUILD.bazel bazel/platforms/BUILD.bazel

bazel-mod-tidy:
	bazelisk mod tidy

bazel-test:
	bazelisk test //:pytest

bazel-test-remote:
	bazelisk test //:pytest --config=remote-gcp-dev

bazel-rbe-smoke:
	scripts/run-bazel-rbe.sh test //:pytest

bazel-check: requirements-lock bazel-format bazel-mod-tidy bazel-test
