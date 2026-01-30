# Quick Start Guide

Get up and running with Kiara SLM in 5 minutes!

## 1. Installation (1 minute)

```bash
# Clone or navigate to project
cd kiara-slm-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

## 2. Prepare Sample Data (1 minute)

```bash
# Create sample training data
cat > data/train.txt << 'EOF'
The development of artificial intelligence has been one of the most significant technological advances.
Machine learning algorithms have revolutionized how we process information and make decisions.
Deep learning uses neural networks with multiple layers to learn from data.
Natural language processing enables computers to understand and generate human language.
The transformer architecture became the foundation for modern language models.
EOF

# Repeat content to have more training data
python -c "
with open('data/train.txt', 'r') as f:
    content = f.read()
with open('data/train.txt', 'w') as f:
    f.write(content * 50)
"
```

## 3. Train a Small Model (2 minutes)

```bash
# Train with small configuration (fast for testing)
python scripts/train.py --config configs/small.yaml
```

This will:
- Create a small GPT model (~10M parameters)
- Train for 10 epochs
- Save checkpoints to `checkpoints/`
- Log progress to console and `logs/`

## 4. Generate Text (30 seconds)

```bash
# Generate text from trained model
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The future of AI" \
    --max-tokens 50
```

## 5. Evaluate Model (30 seconds)

```bash
# Evaluate on validation data
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/train.txt \
    --batch-size 4
```

## Next Steps

### Train a Larger Model

```bash
# Use default configuration (larger model)
python scripts/train.py --config configs/default.yaml
```

### Serve via API

```bash
# Start API server
python scripts/serve.py --checkpoint checkpoints/best_model.pt

# In another terminal, test it:
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Once upon a time", "max_tokens": 50}'
```

### Use Docker

```bash
# Build image
docker build -t kiara-slm:latest .

# Run training
docker-compose up train

# Run API server
docker-compose up api
```

### Customize Configuration

Create your own config file:

```yaml
# configs/my_config.yaml
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

Train with it:

```bash
python scripts/train.py --config configs/my_config.yaml
```

## Common Commands

```bash
# Install with dev dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Clean build artifacts
make clean
```

## Troubleshooting

### Out of Memory

Reduce batch size or model size in config:

```yaml
model:
  emb_dim: 256  # Smaller
  n_layers: 4   # Fewer layers

training:
  batch_size: 2  # Smaller batch
```

### CUDA Not Available

Train on CPU:

```bash
python scripts/train.py --config configs/small.yaml --device cpu
```

### Import Errors

Make sure package is installed:

```bash
pip install -e .
```

## Learn More

- **Full Documentation**: See `README_PRODUCTION.md`
- **Project Structure**: See `PROJECT_STRUCTURE.md`
- **Migration Guide**: See `MIGRATION_GUIDE.md`
- **Tutorial**: See `TUTORIAL.md`

## Getting Help

1. Check documentation files
2. Review example configs in `configs/`
3. Look at test files in `tests/`
4. Check logs in `logs/`

Happy training! 🚀
