# Migration Guide: Old Structure → Production Structure

This guide helps you migrate from the old project structure to the new production-ready structure.

## What Changed?

### Directory Structure

**Old:**
```
src/
├── __init__.py
├── model.py
├── attention.py
├── training.py
└── tokenizer.py
```

**New:**
```
src/kiara/              # Package renamed to 'kiara'
├── __init__.py
├── model.py
├── attention.py
├── training.py
├── tokenizer.py
├── config.py          # NEW: Configuration management
└── utils/             # NEW: Utility modules
    ├── logging.py
    ├── checkpoint.py
    └── metrics.py
```

### Import Changes

**Old imports:**
```python
from src.model import GPTModel
from src.training import train_model_simple
```

**New imports:**
```python
from kiara.model import GPTModel
from kiara.training import train_model_simple
from kiara.config import Config
from kiara.utils import CheckpointManager, setup_logger
```

### Training Script Changes

**Old:**
```bash
python train_quickstart.py
```

**New:**
```bash
# Using configuration file
python scripts/train.py --config configs/small.yaml

# Or using Makefile
make train

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch5_step1000.pt
```

## Step-by-Step Migration

### 1. Update Your Code

If you have custom scripts using the old structure:

```python
# OLD
from src.model import GPTModel
from src.training import train_model_simple

# NEW
from kiara.model import GPTModel
from kiara.training import train_model_simple
```

### 2. Install the Package

```bash
# Install in development mode
pip install -e .

# Or use Makefile
make install
```

### 3. Create Configuration Files

Instead of hardcoding configurations, use YAML files:

```yaml
# configs/my_config.yaml
model:
  vocab_size: 50257
  context_length: 256
  emb_dim: 768
  n_heads: 12
  n_layers: 12
  drop_rate: 0.1

training:
  learning_rate: 5.0e-4
  batch_size: 8
  num_epochs: 10
```

### 4. Use New Training Script

```bash
python scripts/train.py --config configs/my_config.yaml
```

### 5. Update Data Paths

Place your data in the `data/` directory:

```bash
cp your_training_data.txt data/train.txt
cp your_validation_data.txt data/val.txt
```

Update your config:

```yaml
data:
  train_data_path: ./data/train.txt
  val_data_path: ./data/val.txt
```

## New Features Available

### 1. Checkpoint Management

Automatic checkpoint management with best model tracking:

```python
from kiara.utils import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    keep_last_n=3,
    save_best=True
)

# Save checkpoint
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    metrics={'val_loss': val_loss}
)

# Load best checkpoint
checkpoint_manager.load_best_checkpoint(model, optimizer)
```

### 2. Structured Logging

Better logging with file and console output:

```python
from kiara.utils import setup_logger

logger = setup_logger(
    name="my_script",
    log_level="INFO",
    log_dir="./logs",
    log_file="my_script.log"
)

logger.info("Training started")
```

### 3. Configuration Management

Type-safe configuration with validation:

```python
from kiara.config import Config

# Load from YAML
config = Config.from_yaml("configs/default.yaml")

# Load from environment variables
config = load_config_from_env()

# Access configuration
print(config.model.emb_dim)
print(config.training.learning_rate)
```

### 4. Evaluation Metrics

Built-in metrics calculation:

```python
from kiara.utils.metrics import calculate_perplexity, calculate_accuracy

perplexity = calculate_perplexity(loss)
accuracy = calculate_accuracy(logits, targets)
```

### 5. API Server

Serve your model via REST API:

```bash
python scripts/serve.py --checkpoint checkpoints/best_model.pt

# Test it
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 50}'
```

### 6. Docker Support

Run everything in containers:

```bash
# Build image
docker build -t kiara-slm:latest .

# Run training
docker-compose up train

# Run API server
docker-compose up api
```

## Backward Compatibility

The core model and training code remains the same. Your existing:
- Model checkpoints will work
- Training data format is unchanged
- Model architecture is identical

## Getting Help

- See `README_PRODUCTION.md` for detailed production setup
- Check `configs/` for example configurations
- Look at `scripts/` for usage examples
- Run tests with `make test`

## Checklist

- [ ] Update imports in your code
- [ ] Install package with `pip install -e .`
- [ ] Create configuration files
- [ ] Move data to `data/` directory
- [ ] Test with `python scripts/train.py --config configs/small.yaml`
- [ ] Update any custom scripts
- [ ] Run tests with `make test`
- [ ] Review new features in `README_PRODUCTION.md`
