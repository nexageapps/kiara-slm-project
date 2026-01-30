# Kiara SLM - Production Setup Guide

This guide explains the production-ready structure and how to use it.

## Project Structure

```
kiara-slm-project/
├── src/kiara/              # Main package
│   ├── __init__.py
│   ├── model.py           # GPT model architecture
│   ├── attention.py       # Attention mechanisms
│   ├── training.py        # Training utilities
│   ├── tokenizer.py       # Tokenization
│   ├── config.py          # Configuration management
│   └── utils/             # Utility modules
│       ├── logging.py     # Logging setup
│       ├── checkpoint.py  # Checkpoint management
│       └── metrics.py     # Evaluation metrics
│
├── scripts/               # Production scripts
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── generate.py       # Text generation
│   └── serve.py          # API server
│
├── configs/              # Configuration files
│   ├── default.yaml      # Default configuration
│   └── small.yaml        # Small model config
│
├── tests/                # Unit tests
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_config.py
│   └── test_utils.py
│
├── data/                 # Training data
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── notebooks/            # Jupyter notebooks
│
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── Makefile             # Build automation
├── setup.py             # Package setup
├── pyproject.toml       # Project metadata
├── requirements.txt     # Dependencies
└── .env.example         # Environment template
```

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
make install

# Or install with dev dependencies
make install-dev
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Or use YAML configs in configs/
```

### 3. Prepare Data

```bash
# Place your training data in data/
# Format: plain text files
cp your_data.txt data/train.txt
```

### 4. Training

```bash
# Using Makefile
make train

# Or directly with Python
python scripts/train.py --config configs/small.yaml

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch5_step1000.pt
```

### 5. Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/val.txt \
    --batch-size 8
```

### 6. Text Generation

```bash
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --max-tokens 100 \
    --temperature 0.8
```

### 7. API Server

```bash
python scripts/serve.py \
    --checkpoint checkpoints/best_model.pt \
    --host 0.0.0.0 \
    --port 8000

# Test the API
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The future of AI", "max_tokens": 50}'
```

## Docker Usage

### Build Image

```bash
make docker-build
# Or: docker build -t kiara-slm:latest .
```

### Run with Docker Compose

```bash
# Training
docker-compose up train

# Evaluation
docker-compose up evaluate

# API Server
docker-compose up api
```

## Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Configuration Management

### Using YAML Files

```yaml
# configs/custom.yaml
model:
  vocab_size: 50257
  context_length: 512
  emb_dim: 1024
  n_heads: 16
  n_layers: 24

training:
  learning_rate: 3.0e-4
  batch_size: 16
  num_epochs: 20
```

```bash
python scripts/train.py --config configs/custom.yaml
```

### Using Environment Variables

```bash
# Set in .env or export
export MODEL_SIZE=medium
export BATCH_SIZE=16
export NUM_EPOCHS=20
export USE_WANDB=true

python scripts/train.py
```

## Checkpoint Management

Checkpoints are automatically managed:
- Saves every N steps (configurable)
- Keeps last N checkpoints (default: 3)
- Always saves best model based on validation loss
- Stored in `checkpoints/` directory

## Logging

Logs are saved to:
- Console: INFO level
- File: DEBUG level in `logs/` directory
- Optional: Weights & Biases integration

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Production Deployment

### Using Docker

```bash
# Build production image
docker build -t kiara-slm:prod .

# Run API server
docker run -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    kiara-slm:prod python scripts/serve.py \
    --checkpoint /app/checkpoints/best_model.pt
```

### Using Kubernetes

See `k8s/` directory for Kubernetes manifests (to be added).

## Monitoring

- Training metrics logged to console and files
- Optional Weights & Biases integration
- API server provides `/health` endpoint

## Best Practices

1. **Always use configuration files** for reproducibility
2. **Version your configs** alongside checkpoints
3. **Monitor validation loss** to prevent overfitting
4. **Use mixed precision** for faster training (enabled by default)
5. **Save checkpoints frequently** to prevent data loss
6. **Test on small data** before full training runs

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Reduce `context_length`
- Use smaller model size
- Enable gradient checkpointing (to be added)

### Slow Training

- Increase `batch_size` if memory allows
- Use mixed precision training
- Reduce `eval_freq` for less frequent evaluation
- Use multiple workers for data loading

### Poor Generation Quality

- Train for more epochs
- Increase model size
- Use more training data
- Adjust temperature and top_k during generation

## Contributing

See main README.md for contribution guidelines.

## License

MIT License - see LICENSE file for details.
