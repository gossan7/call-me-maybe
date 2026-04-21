VENV        := .venv
POETRY      := $(VENV)/bin/poetry
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
SRC			:= ./src
ARGS		?= ./maps/customs/priority_optimization.txt

$(VENV):
	python3 -m venv $(VENV)

$(POETRY): | $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install poetry

.PHONY: install run debug clean lint lint-strict

install: $(POETRY)
	$(POETRY) install

run: install
	$(POETRY) run python $(SRC)/main.py $(ARGS)

debug: install
	$(POETRY) run python -m pdb $(SRC)/main.py $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

fclean: clean
	rm -rf $(VENV)

lint: install
	$(POETRY) run flake8 . --exclude .venv
	$(POETRY) run mypy . --exclude .venv --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	$(POETRY) run flake8 . --exclude .venv
	$(POETRY) run mypy . --exclude .venv --strict