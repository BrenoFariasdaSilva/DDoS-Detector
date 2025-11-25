# Variables
VENV := venv
OS := $(shell uname 2>/dev/null || echo Windows)

# Detect correct Python and Pip commands based on OS
ifeq ($(OS), Windows)
	PYTHON := $(VENV)/Scripts/python.exe
	PIP := $(VENV)/Scripts/pip.exe
	PYTHON_CMD := python
	CLEAR_CMD := cls
	TIME_CMD :=
else
	PYTHON := $(VENV)/bin/python3
	PIP := $(VENV)/bin/pip
	PYTHON_CMD := python3
	CLEAR_CMD := clear
	TIME_CMD := time
endif

# Default target
all: main

# Run main scripts
main: $(VENV)
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./main.py

dataset_descriptor: $(VENV)
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./dataset_descriptor.py

genetic_algorithm: $(VENV)
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./genetic_algorithm.py

rfe: $(VENV)
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./rfe.py

telegram: $(VENV)
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./telegram_bot.py

# Create virtual environment and install dependencies
$(VENV):
	@echo "Creating virtual environment..."
	$(PYTHON_CMD) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Install dependencies
dependencies: $(VENV)

# Generate requirements.txt from current venv
generate_requirements: $(VENV)
	$(PIP) freeze > requirements.txt

# Clean artifacts
clean:
	rm -rf $(VENV) || rmdir /S /Q $(VENV) 2>nul
	find . -type f -name '*.pyc' -delete || del /S /Q *.pyc 2>nul
	find . -type d -name '__pycache__' -delete || rmdir /S /Q __pycache__ 2>nul

.PHONY: all main clean dependencies generate_requirements dataset_descriptor genetic_algorithm rfe telegram