# Variables
VENV := venv
OS := $(shell uname 2>/dev/null || echo Windows)

# Logs directory
LOG_DIR := ./Logs

# Ensure logs directory exists (cross-platform)
ENSURE_LOG_DIR := @mkdir -p $(LOG_DIR) 2>/dev/null || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"

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
main: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./main.py)

dataset_converter: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./dataset_converter.py)

dataset_descriptor: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./dataset_descriptor.py)

hyperparameters_optimization: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./hyperparameters_optimization.py)

genetic_algorithm: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./genetic_algorithm.py)

pca: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./pca.py)

rfe: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./rfe.py)

stacking: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./stacking.py)

telegram: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./telegram_bot.py)

wgangp: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	$(call RUN_AND_LOG, ./wgangp.py)

# Create virtual environment if missing
$(VENV):
	@echo "Creating virtual environment..."
	$(PYTHON_CMD) -m venv $(VENV)
	$(PIP) install --upgrade pip

dependencies: $(VENV)
	@echo "Installing/Updating Python dependencies..."
	$(PIP) install -r requirements.txt

# Generate requirements.txt from current venv
generate_requirements: $(VENV)
	$(PIP) freeze > requirements.txt

# Clean artifacts
clean:
	rm -rf $(VENV) || rmdir /S /Q $(VENV) 2>nul
	find . -type f -name '*.pyc' -delete || del /S /Q *.pyc 2>nul
	find . -type d -name '__pycache__' -delete || rmdir /S /Q __pycache__ 2>nul

.PHONY: all main clean dependencies generate_requirements dataset_converter dataset_descriptor genetic_algorithm hyperparameters_optimization pca rfe stacking telegram wgangp