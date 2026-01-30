# Kiara SLM - Production Project Structure

## Complete Directory Tree

```
kiara-slm-project/
│
├── src/kiara/                      # Main Python package
│   ├── __init__.py                # Package initialization
│   ├── model.py                   # GPT model architecture
│   ├── attention.py               # Attention mechanisms
│   ├── training.py                # Training utilities & data loaders
│   ├── tokenizer.py               # Tokenization utilities
│   ├── config.py                  # Configuration management
│   │
│   ├── cli/                       # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── train.py              # Training CLI entry point
│   │   ├── evaluate.py           # Evaluation CLI entry point
│   │   └── generate.py           # Generation CLI entry point
│   │
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       ├── logging.py            # Logging configuration
│       ├── checkpoint.py         # Checkpoint management
│       └── metrics.py            # Evaluation metrics
│
├── scripts/                       # Production scripts
│   ├── __init__.py
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Model evaluation script
│   ├── generate.py               # Text generation script
│   └── serve.py                  # API server (FastAPI)
│
├── configs/                       # Configuration files
│   ├── default.yaml              # Default configuration
│   └── small.yaml                # Small model config (for testing)
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_model.py             # Model architecture tests
│   ├── test_training.py          # Training utilities tests
│   ├── test_config.py            # Configuration tests
│   └── test_utils.py             # Utility function tests
│
├── data/                          # Training data directory
│   ├── .gitkeep
│   ├── train.txt                 # Training data (not in git)
│   └── val.txt                   # Validation data (not in git)
│
├── checkpoints/                   # Model checkpoints
│   ├── .gitkeep
│   ├── checkpoint_epoch*.pt      # Training checkpoints (not in git)
│   └── best_model.pt             # Best model checkpoint (not in git)
│
├── logs/                          # Training logs
│   ├── .gitkeep
│   └── *.log                     # Log files (not in git)
│
├── notebooks/                     # Jupyter notebooks
│   └── getting_started.md        # Getting started guide
│
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker Compose configuration
├── Makefile                       # Build automation
│
├── setup.py                       # Package installation script
├── pyproject.toml                 # Project metadata & tool configs
├── requirements.txt               # Python dependencies
│
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── .gitattributes                 # Git attributes
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
│
├── README.md                      # Main project documentation
├── README_PRODUCTION.md           # Production setup guide
├── MIGRATION_GUIDE.md             # Migration from old structure
├── PROJECT_STRUCTURE.md           # This file
└── TUTORIAL.md                    # Tutorial documentation
```

## Key Components

### 1. Source Code (`src/kiara/`)

**Core Modules:**
- `model.py`: GPT architecture implementation
- `attention.py`: Self-attention and multi-head attention
- `training.py`: Training loop, data loaders, generation
- `tokenizer.py`: Text tokenization utilities
- `config.py`: Configuration management with dataclasses

**Utilities (`utils/`):**
- `logging.py`: Structured logging setup
- `checkpoint.py`: Automatic checkpoint management
- `metrics.py`: Evaluation metrics (perplexity, accuracy)

**CLI (`cli/`):**
- Entry points for command-line tools
- Installed as console scripts via setup.py

### 2. Scripts (`scripts/`)

Production-ready scripts for common tasks:

- **train.py**: Full-featured training with:
  - Configuration file support
  - Checkpoint resumption
  - Automatic logging
  - Validation monitoring
  
- **evaluate.py**: Model evaluation with:
  - Loss calculation
  - Perplexity metrics
  - Accuracy measurement
  
- **generate.py**: Text generation with:
  - Greedy decoding
  - Temperature sampling
  - Top-k filtering
  
- **serve.py**: REST API server with:
  - FastAPI framework
  - Health check endpoint
  - Text generation endpoint

### 3. Configuration (`configs/`)

YAML configuration files for different scenarios:

- **default.yaml**: Standard configuration
- **small.yaml**: Lightweight config for testing

Configuration structure:
```yaml
model:          # Model architecture
training:       # Training hyperparameters
data:           # Data paths and settings
logging:        # Logging configuration
```

### 4. Tests (`tests/`)

Comprehensive test suite:

- **test_model.py**: Model architecture tests
- **test_training.py**: Training utilities tests
- **test_config.py**: Configuration management tests
- **test_utils.py**: Utility function tests

Run with: `make test` or `pytest`

### 5. Docker Support

**Dockerfile:**
- Multi-stage build
- Python 3.10 base
- GPU support ready

**docker-compose.yml:**
- Training service
- Evaluation service
- API server service

### 6. Build Tools

**Makefile:**
- `make install`: Install package
- `make test`: Run tests
- `make lint`: Code quality checks
- `make format`: Auto-format code
- `make train`: Run training
- `make docker-build`: Build Docker image

**setup.py:**
- Package installation
- Console script entry points
- Dependency management

**pyproject.toml:**
- Tool configurations (black, isort, pytest, mypy)
- Package metadata

### 7. Development Tools

**.pre-commit-config.yaml:**
- Automatic code formatting
- Linting checks
- YAML/JSON validation

**.gitignore:**
- Excludes checkpoints, logs, data
- Python artifacts
- IDE files

## Data Flow

```
1. Data Preparation
   data/train.txt → GPTDataset → DataLoader

2. Training
   DataLoader → GPTModel → Loss → Optimizer → Checkpoints

3. Evaluation
   Checkpoint → GPTModel → Metrics (loss, perplexity, accuracy)

4. Generation
   Checkpoint → GPTModel → Generated Text

5. Serving
   Checkpoint → GPTModel → FastAPI → REST API
```

## Configuration Flow

```
1. Environment Variables (.env)
   ↓
2. YAML Config Files (configs/*.yaml)
   ↓
3. Config Dataclasses (config.py)
   ↓
4. Training/Evaluation Scripts
```

## Checkpoint Management

```
Training → CheckpointManager
           ├── checkpoint_epoch1_step1000.pt
           ├── checkpoint_epoch2_step2000.pt
           ├── checkpoint_epoch3_step3000.pt  (keeps last N)
           └── best_model.pt  (best validation loss)
```

## Logging Flow

```
Training Events
    ↓
Logger (logging.py)
    ├── Console Output (INFO level)
    ├── File Output (DEBUG level)
    └── Optional: Weights & Biases
```

## API Architecture

```
Client Request
    ↓
FastAPI (serve.py)
    ↓
GPTModel (loaded from checkpoint)
    ↓
Text Generation
    ↓
JSON Response
```

## Development Workflow

1. **Setup**: `make install-dev`
2. **Configure**: Edit `configs/` or `.env`
3. **Develop**: Write code in `src/kiara/`
4. **Test**: `make test`
5. **Format**: `make format`
6. **Train**: `make train`
7. **Evaluate**: `python scripts/evaluate.py`
8. **Deploy**: `docker-compose up`

## Production Deployment

1. **Build**: `docker build -t kiara-slm:prod .`
2. **Configure**: Set environment variables
3. **Train**: `docker-compose up train`
4. **Serve**: `docker-compose up api`
5. **Monitor**: Check logs in `logs/`

## Best Practices

1. **Always use configs**: Don't hardcode parameters
2. **Version checkpoints**: Save config with each checkpoint
3. **Monitor validation**: Watch for overfitting
4. **Test before training**: Use small config first
5. **Use Docker**: For reproducible environments
6. **Write tests**: For custom modifications
7. **Log everything**: Use structured logging

## Extension Points

To extend the project:

1. **New model architectures**: Add to `src/kiara/model.py`
2. **Custom training loops**: Extend `src/kiara/training.py`
3. **New metrics**: Add to `src/kiara/utils/metrics.py`
4. **API endpoints**: Extend `scripts/serve.py`
5. **Data preprocessing**: Add to `src/kiara/tokenizer.py`

## File Naming Conventions

- **Python modules**: lowercase with underscores (`model.py`)
- **Classes**: PascalCase (`GPTModel`)
- **Functions**: lowercase with underscores (`train_model_simple`)
- **Constants**: UPPERCASE (`GPT_CONFIG`)
- **Config files**: lowercase with extension (`.yaml`, `.json`)
- **Checkpoints**: descriptive with metadata (`checkpoint_epoch5_step1000.pt`)

## Import Conventions

```python
# Standard library
import os
from pathlib import Path

# Third-party
import torch
import numpy as np

# Local package
from kiara.model import GPTModel
from kiara.config import Config
from kiara.utils import setup_logger
```

## Documentation

- **README.md**: Project overview and quick start
- **README_PRODUCTION.md**: Detailed production guide
- **MIGRATION_GUIDE.md**: Upgrade from old structure
- **PROJECT_STRUCTURE.md**: This file
- **TUTORIAL.md**: Step-by-step tutorial
- **Docstrings**: In all modules and functions

## Support

For questions or issues:
1. Check documentation files
2. Review example configs
3. Run tests to verify setup
4. Check logs for errors
5. Open GitHub issue
