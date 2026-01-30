# Configuration Guide

Complete guide to configuring Kiara SLM.

## Configuration Methods

### 1. YAML Files (Recommended)

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
  weight_decay: 0.1
  warmup_steps: 100
  gradient_clip: 1.0
  eval_freq: 100
  eval_iter: 10
  save_every_n_steps: 1000
  keep_last_n_checkpoints: 3

data:
  data_dir: ./data
  train_data_path: ./data/train.txt
  val_data_path: ./data/val.txt
  stride: 128
  num_workers: 4

logging:
  log_level: INFO
  log_dir: ./logs
  use_wandb: false
  wandb_project: kiara-slm
  wandb_entity: null

checkpoint_dir: ./checkpoints
device: cuda
mixed_precision: true
seed: 42
```

### 2. Environment Variables

```bash
# .env file
MODEL_SIZE=small
CONTEXT_LENGTH=256
BATCH_SIZE=8
LEARNING_RATE=5e-4
NUM_EPOCHS=10

DATA_DIR=./data
TRAIN_DATA_PATH=./data/train.txt
VAL_DATA_PATH=./data/val.txt

LOG_LEVEL=INFO
USE_WANDB=false
WANDB_PROJECT=kiara-slm

DEVICE=cuda
MIXED_PRECISION=true
```

### 3. Command Line Arguments

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --device cuda \
    --resume checkpoints/checkpoint_epoch5_step1000.pt
```

## Configuration Sections

### Model Configuration

Controls model architecture:

```yaml
model:
  vocab_size: 50257        # Vocabulary size (GPT-2: 50257)
  context_length: 256      # Maximum sequence length
  emb_dim: 768            # Embedding dimension
  n_heads: 12             # Number of attention heads
  n_layers: 12            # Number of transformer blocks
  drop_rate: 0.1          # Dropout rate
```

**Preset Sizes:**

**Small** (Fast training, ~10M params):
```yaml
emb_dim: 256
n_heads: 4
n_layers: 4
```

**Medium** (~100M params):
```yaml
emb_dim: 1024
n_heads: 16
n_layers: 24
```

**Large** (~300M params):
```yaml
emb_dim: 1280
n_heads: 20
n_layers: 36
```

### Training Configuration

Controls training process:

```yaml
training:
  learning_rate: 5.0e-4           # Learning rate
  batch_size: 8                   # Batch size
  num_epochs: 10                  # Number of epochs
  weight_decay: 0.1               # Weight decay (L2 regularization)
  warmup_steps: 100               # Learning rate warmup steps
  gradient_clip: 1.0              # Gradient clipping threshold
  eval_freq: 100                  # Evaluate every N steps
  eval_iter: 10                   # Number of batches for evaluation
  save_every_n_steps: 1000        # Save checkpoint every N steps
  keep_last_n_checkpoints: 3      # Number of checkpoints to keep
```

### Data Configuration

Controls data loading:

```yaml
data:
  data_dir: ./data                    # Data directory
  train_data_path: ./data/train.txt   # Training data file
  val_data_path: ./data/val.txt       # Validation data file
  stride: 128                         # Sliding window stride
  num_workers: 4                      # Data loader workers
```

### Logging Configuration

Controls logging behavior:

```yaml
logging:
  log_level: INFO              # Logging level (DEBUG, INFO, WARNING, ERROR)
  log_dir: ./logs             # Log directory
  use_wandb: false            # Enable Weights & Biases
  wandb_project: kiara-slm    # W&B project name
  wandb_entity: null          # W&B entity/username
```

### Global Configuration

```yaml
checkpoint_dir: ./checkpoints   # Checkpoint directory
device: cuda                    # Device (cuda/cpu)
mixed_precision: true           # Enable mixed precision training
seed: 42                        # Random seed
```

## Configuration Priority

When multiple configuration sources are used:

1. Command line arguments (highest priority)
2. Environment variables
3. YAML configuration file
4. Default values (lowest priority)

## Example Configurations

### Quick Testing

```yaml
# configs/test.yaml
model:
  emb_dim: 128
  n_heads: 4
  n_layers: 2
  context_length: 64

training:
  batch_size: 2
  num_epochs: 2
  eval_freq: 10
```

### Production Training

```yaml
# configs/production.yaml
model:
  emb_dim: 1024
  n_heads: 16
  n_layers: 24
  context_length: 1024

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 3.0e-4
  save_every_n_steps: 500
  keep_last_n_checkpoints: 5

logging:
  use_wandb: true
  wandb_project: kiara-slm-production
```

### CPU Training

```yaml
# configs/cpu.yaml
model:
  emb_dim: 256
  n_heads: 4
  n_layers: 4

training:
  batch_size: 2

device: cpu
mixed_precision: false
```

## Using Configurations

### In Python

```python
from kiara.config import Config

# Load from YAML
config = Config.from_yaml("configs/default.yaml")

# Load from environment
from kiara.config import load_config_from_env
config = load_config_from_env()

# Access values
print(config.model.emb_dim)
print(config.training.learning_rate)

# Save configuration
config.save("configs/my_config.yaml")
```

### In Scripts

```bash
# Use specific config
python scripts/train.py --config configs/small.yaml

# Override device
python scripts/train.py --config configs/default.yaml --device cpu

# Resume training
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch5_step1000.pt
```

## Best Practices

1. **Version control configs**: Keep configs in git
2. **Document changes**: Comment why you changed values
3. **Start small**: Test with small config first
4. **Save with checkpoints**: Store config alongside model
5. **Use presets**: Start from provided configs
6. **Monitor metrics**: Adjust based on training curves
7. **Reproducibility**: Set seed for reproducible results

## Troubleshooting

### Out of Memory

Reduce:
- `batch_size`
- `context_length`
- `emb_dim`
- `n_layers`

### Slow Training

Increase:
- `batch_size` (if memory allows)
- Reduce `eval_freq`
- Increase `num_workers`

### Poor Convergence

Adjust:
- `learning_rate` (try 1e-4 to 5e-4)
- `warmup_steps` (increase for stability)
- `weight_decay` (reduce if underfitting)

### Overfitting

Increase:
- `drop_rate`
- `weight_decay`
- Training data size
