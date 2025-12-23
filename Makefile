# Variables
VENV := venv
OS := $(shell uname 2>/dev/null || echo Windows)

# Detect correct Python and Pip commands based on OS
ifeq ($(OS), Windows) # Windows
	PYTHON := $(VENV)/Scripts/python.exe
	PIP := $(VENV)/Scripts/pip.exe
	PYTHON_CMD := python
	CLEAR_CMD := cls
	TIME_CMD :=
else # Unix-like
	PYTHON := $(VENV)/bin/python3
	PIP := $(VENV)/bin/pip
	PYTHON_CMD := python3
	CLEAR_CMD := clear
	TIME_CMD := time
endif

# Logs directory
LOG_DIR := ./Logs

# Ensure logs directory exists (cross-platform)
ENSURE_LOG_DIR := @mkdir -p $(LOG_DIR) 2>/dev/null || $(PYTHON_CMD) -c "import os; os.makedirs('$(LOG_DIR)', exist_ok=True)"

# Run-and-log function
# On Windows: simply runs the Python script normally
# On Unix-like systems: supports DETACH variable
#   - If DETACH is set, runs the script in detached mode and tails the log file
#   - Else, runs the script normally
ifeq ($(OS), Windows) # Windows
RUN_AND_LOG = $(PYTHON) $(1)
else
RUN_AND_LOG = \
if [ -z "$(DETACH)" ]; then \
	$(PYTHON) $(1); \
else \
	LOG_FILE=$(LOG_DIR)/$$(basename $(1) .py).log; \
	nohup $(PYTHON) $(1) --verbose >/dev/null 2>&1 & \
	sleep 2; \
	tail -f $$LOG_FILE; \
fi
endif

# Default target
all: main

# Make Rules
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

download_datasets: dependencies
	$(ENSURE_LOG_DIR)
	$(CLEAR_CMD)
	chmod +x ./download_datasets.sh || ;
	./download_datasets.sh

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

# Run code quality checks: syntax, lint, and type checking
check-build: dependencies
	@echo "Running syntax check, flake8 linting, and mypy type checks..."
	# Ensure linters are installed in the venv
	$(PIP) install --upgrade flake8 mypy
	# Python syntax check (compile all .py files, excluding venv/.git/.github/.assets)
	find . -name "*.py" \
		-not -path "./venv/*" -not -path "./.venv/*" -not -path "./.git/*" -not -path "./.github/*" -not -path "./.assets/*" -print0 | xargs -0 -n1 $(PYTHON) -m py_compile
	# Run flake8 (excluding common directories)
	$(PYTHON) -m flake8 . --max-line-length=120 --exclude=.git,venv,.venv,.github,.assets
	# Run mypy (exclude via regex)
	$(PYTHON) -m mypy . --exclude '(^\.venv/|^venv/|^\.git/|^\.github/|^\.assets/)'

# Auto-fix Python style issues using ruff and black, then normalize tabs -> spaces
fix-style: dependencies
	@echo "Auto-fixing style with ruff and black, converting tabs to spaces..."
	# Ensure formatters are installed in the venv
	$(PIP) install --upgrade ruff black
	# Run ruff auto-fixes (exclude large/non-source dirs)
	$(PYTHON) -m ruff --fix . --extend-exclude venv,.venv,.git,.github,.assets || true
	# Run black to reformat and wrap long lines to the project's max length
	$(PYTHON) -m black . --line-length 120 --exclude 'venv|\.venv|\.git|\.github|\.assets' || true
	# Convert any remaining tabs to 4-space indentation for .py files (excluding venv/.git/.github/.assets)
	find . -name "*.py" \
		-not -path "./venv/*" -not -path "./.venv/*" -not -path "./.git/*" -not -path "./.github/*" -not -path "./.assets/*" -print0 \
		| xargs -0 -I {} sh -c 'expand -t 4 "{}" > "{}.exp" && mv "{}.exp" "{}"' || true

# Generate requirements.txt from current venv
generate_requirements: $(VENV)
	$(PIP) freeze > requirements.txt

# Clean artifacts
clean:
	rm -rf $(VENV) || rmdir /S /Q $(VENV) 2>nul
	find . -type f -name '*.pyc' -delete || del /S /Q *.pyc 2>nul
	find . -type d -name '__pycache__' -delete || rmdir /S /Q __pycache__ 2>nul

.PHONY: all main clean dependencies check-build fix-style generate_requirements dataset_converter dataset_descriptor download_datasets genetic_algorithm hyperparameters_optimization pca rfe stacking telegram wgangp