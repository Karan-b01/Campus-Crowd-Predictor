.PHONY: setup lint test audit train app

ifeq ($(OS),Windows_NT)
SHELL = cmd.exe
PYTHON = $(CURDIR)/.venv/Scripts/python.exe
else
PYTHON = $(CURDIR)/.venv/bin/python
endif

setup:
	python -m venv --without-pip .venv
	python -m pip --python "$(PYTHON)" install -r requirements.txt

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) -m src.audit

train:
	$(PYTHON) -m src.train

app:
	$(PYTHON) -m streamlit run app/main.py
