# Variables
VENV := venv
OS := $(shell uname 2>/dev/null || echo Windows)

# Logs directory
LOG_DIR := ./logs

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
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./main.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./main.py)).log

dataset_descriptor: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./dataset_descriptor.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./dataset_descriptor.py)).log

hyperparameters_optimization: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./hyperparameters_optimization.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./hyperparameters_optimization.py)).log

genetic_algorithm: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./genetic_algorithm.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./genetic_algorithm.py)).log

pca: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./pca.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./pca.py)).log

rfe: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./rfe.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./rfe.py)).log

stacking: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./stacking.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./stacking.py)).log

telegram: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./telegram_bot.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./telegram_bot.py)).log

wgangp: dependencies
	@mkdir -p $(LOG_DIR) 2>nul || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"
	$(CLEAR_CMD)
	$(TIME_CMD) $(PYTHON) ./wgangp.py 2>&1 | tee $(LOG_DIR)/$(notdir $(basename ./wgangp.py)).log

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