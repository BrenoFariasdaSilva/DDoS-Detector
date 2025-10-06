# Detect system Python dynamically
PYTHON := $(shell command -v python3 2>/dev/null || command -v python)
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# For Windows compatibility
ifeq ($(OS),Windows_NT)
	PIP := $(VENV)/Scripts/pip.exe
	PYTHON_VENV := $(VENV)/Scripts/python.exe
endif

# Default target
all: run

# Run main scripts
main: $(VENV)
	clear;
	time $(PYTHON_VENV) ./main.py

# Create virtual environment and install dependencies
$(VENV):
	@echo "Using Python at: $(PYTHON)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Install dependencies
dependencies: $(VENV)

# Generate requirements.txt from current venv
generate_requirements: $(VENV)
	$(PIP) freeze > requirements.txt

# Clean artifacts
clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

.PHONY: main clean dependencies generate_requirements
