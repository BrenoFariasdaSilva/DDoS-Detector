# Variables
VENV := venv
OS := $(shell uname 2>/dev/null || echo Windows)

# Logs directory
LOG_DIR := ./Logs

# Ensure logs directory exists (cross-platform). Use as: $(ENSURE_LOG_DIR)
ENSURE_LOG_DIR := @mkdir -p $(LOG_DIR) 2>/dev/null || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"

# Reusable run-and-log function: call with the script path, e.g. $(call RUN_AND_LOG, ./script.py)
ifeq ($(OS), Windows)
RUN_AND_LOG = (start /B $(PYTHON) $(1) 2>&1 | tee $(LOG_DIR)/$(notdir $(basename $(1))).raw.log) ; \
					sed -r "s/\x1B\[[0-9;]*[a-zA-Z]//g" \
						$(LOG_DIR)/$(notdir $(basename $(1))).raw.log \
						> $(LOG_DIR)/$(notdir $(basename $(1))).log ; \
					rm $(LOG_DIR)/$(notdir $(basename $(1))).raw.log
else
RUN_AND_LOG = (nohup $(TIME_CMD) $(PYTHON) $(1) > $(LOG_DIR)/$(notdir $(basename $(1))).raw.log 2>&1 & \
					tail -f $(LOG_DIR)/$(notdir $(basename $(1))).raw.log | sed -r "s/\x1B\[[0-9;]*[a-zA-Z]//g" \
						> $(LOG_DIR)/$(notdir $(basename $(1))).log ; \
					rm $(LOG_DIR)/$(notdir $(basename $(1))).raw.log)
endif

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

.PHONY: all main clean dependencies generate_requirements dataset_descriptor genetic_algorithm hyperparameters_optimization pca rfe stacking telegram wgangp