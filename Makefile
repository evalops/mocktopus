
.PHONY: fmt test

fmt:
	ruff check --fix || true

test:
	pytest -q
