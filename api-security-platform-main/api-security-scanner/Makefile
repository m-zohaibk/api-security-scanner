# Makefile with common convenience targets
.PHONY: preflight test test-all lint run-dashboard

preflight:
	python tools/check_environment.py

# Fast test suite used during development
test:
	pytest -q tests/test_core.py tests/test_feature_vector.py tests/test_setup.py tests/test_dashboard.py tests/test_database.py tests/test_preflight.py

# Run the full test suite
test-all:
	pytest -q

# Run linters/formatters if installed; non-fatal if tools missing
lint:
	@echo "Running linters/formatters if installed"
	@ruff check . || true
	@flake8 . || true
	@black --check . || true

# Launch the web dashboard in foreground (unbuffered)
run-dashboard:
	python -u main.py --dashboard
