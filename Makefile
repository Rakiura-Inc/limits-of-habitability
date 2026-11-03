# Local work, all through uv (https://docs.astral.sh/uv/getting-started/installation/).
#   make local   install the package with both projects' extras and the dev tools, then open Jupyter Lab
#   make test    run the tests
#   make lint    run every git hook (ruff, nbstripout and the rest) on every file
#   make data    download the archive files into notebooks/data/ and check them against their checksums
#   make run     execute every notebook headless (blank answers, offline) into notebooks/executed/

RAKIURA_HELPER ?= rakiura @ git+https://github.com/Rakiura-Inc/rakiura-py@main
NOTEBOOKS := $(wildcard notebooks/*.ipynb)
# Seconds one cell may run before make run stops. CI sets it lower, so a stuck cell fails fast and is named.
CELL_TIMEOUT ?= 3600

.PHONY: local install test lint data run clean

local: install
	uv run --all-extras jupyter lab notebooks

install:
	uv sync --all-extras
	@uv pip install -q "$(RAKIURA_HELPER)" && echo "Installed the Rakiura helper." \
		|| echo "Could not install the Rakiura helper, so the notebooks run offline."

test:
	uv run --all-extras pytest

lint:
	uvx prek run --all-files

data:
	uv run --all-extras python scripts/fetch_archive_files.py

run: data
	@mkdir -p notebooks/executed
	@for nb in $(NOTEBOOKS); do \
		echo "running $$nb"; \
		RAKIURA_LIVE=0 uv run --all-extras jupyter nbconvert --to notebook --execute \
			--ExecutePreprocessor.timeout=$(CELL_TIMEOUT) --output-dir notebooks/executed "$$nb" || exit 1; \
	done

clean:
	rm -rf .venv dist handin data notebooks/executed notebooks/handin notebooks/data notebooks/iota_dra_rv.csv
