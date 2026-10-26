# Local testing.
#   make local      create .venv, install this package (editable) and the Rakiura helper, then open Jupyter Lab
#   make test       run the tests
#   make run        execute every notebook headless (blank answers) into notebooks/executed/

PYTHON ?= python3
VENV ?= .venv
RAKIURA_HELPER ?= rakiura @ git+https://github.com/Rakiura-Inc/rakiura-py@main
BIN := $(VENV)/bin

.PHONY: local venv install test run clean

local: install
	$(BIN)/jupyter lab notebooks

venv:
	test -x $(BIN)/python || $(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -q --upgrade pip

install: venv
	$(BIN)/pip install -q -e ".[dev,notebooks]"
	@$(BIN)/pip install -q "$(RAKIURA_HELPER)" && echo "Installed the Rakiura helper." \
		|| echo "Could not install the Rakiura helper, so the notebooks run offline."

test: install
	$(BIN)/python -m pytest -q

run: install
	@mkdir -p notebooks/executed
	@for nb in notebooks/0*.ipynb; do \
		echo "running $$nb"; \
		RAKIURA_LIVE=0 $(BIN)/jupyter nbconvert --to notebook --execute --output-dir notebooks/executed "$$nb" || exit 1; \
	done

clean:
	rm -rf $(VENV) handin data
