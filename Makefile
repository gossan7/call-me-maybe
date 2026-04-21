SRC         := src
ARGS        ?=

.PHONY: install run debug clean fclean lint lint-strict

install:
	uv sync

run: install
	uv run python -m $(SRC) $(ARGS)

debug: install
	uv run python -m pdb -m $(SRC) $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

fclean: clean
	rm -rf .venv

lint: install
	uv run flake8 . --exclude .venv,llm_sdk
	uv run mypy . --exclude '(.venv|llm_sdk)' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run flake8 . --exclude .venv,llm_sdk
	uv run mypy . --exclude '(.venv|llm_sdk)' --strict
