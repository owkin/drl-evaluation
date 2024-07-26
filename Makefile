.PHONY: clean clean-test clean-pyc clean-build help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

ifeq (, $(shell which snakeviz))
	PROFILE = pytest --profile-svg
	PROFILE_RESULT = prof/combined.svg
	PROFILE_VIEWER = $(BROWSER)
else
    PROFILE = pytest --profile
    PROFILE_RESULT = prof/combined.prof
	PROFILE_VIEWER = snakeviz
endif

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

lock: ## generate a new poetry.lock file (To be done after adding new requirements to pyproject.toml)
	poetry lock

install: clean ## install all package dependencies to the active Python's site-packages
	poetry config certificates.owkin.cert false ## temporary until SSL certif is fixed
	curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	. "${HOME}/.cargo/env"
	poetry install -E gnn

install-M1: clean ## install all package dependencies to the active Python's site-packages
	curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	. "${HOME}/.cargo/env"
	poetry install

install-all: install ## install all package and development dependencies for testing to the active Python's site-packages
	poetry install --with=testing,linting --no-root -E gnn

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -path ./.venv -prune -false -o -name '*.egg-info' -exec rm -fr {} +
	find . -path ./.venv -prune -false -o -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -path ./.venv -prune -false -o -name '*.pyc' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*.pyo' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*~' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr prof/

format: ## format code by sorting imports with black
	docformatter omics_rpz tests --in-place
	isort omics_rpz tests
	black omics_rpz tests

lint: ## check style with pylint
	-flake8 omics_rpz tests
	pylint omics_rpz tests


typing: ## check static typing using mypy
	mypy omics_rpz

pre-commit-checks: ## Run pre-commit checks on all files
	pre-commit run --hook-stage manual --all-files

lint-all: pre-commit-checks lint typing ## Run all linting checks.

test: ## run tests quickly with the default Python
	pytest

coverage: ## check code coverage quickly with the default Python
	coverage run --source omics_rpz -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

profile:  ## create a profile from test cases
	$(PROFILE) $(TARGET)
	$(PROFILE_VIEWER) $(PROFILE_RESULT)
