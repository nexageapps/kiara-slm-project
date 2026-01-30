# Technical Documentation

This document provides an overview of the Kiara SLM project and explains the purpose and role of the main files and modules.

---

## Project Overview

Kiara is a Small Language Model (SLM) implementation built from scratch. It is a **decoder-only transformer** (GPT-style) that learns to predict the next token in a sequence. The codebase is organized into:

- **Core library** (`src/kiara/`): Model, training, config, and utilities
- **Scripts** (`scripts/`): Entry points for training, evaluation, generation, and serving
- **Configuration** (`configs/`): YAML configs for different model sizes and runs
- **Tests** (`tests/`): Unit tests for the core components

Training reads plain text from `data/train.txt` (and `data/val.txt` for validation), tokenizes it, and updates the model via next-token prediction. Trained weights are saved as PyTorch checkpoints in `checkpoints/`. No database is used; the model’s “memory” is entirely in those checkpoint files.

---

## Main Directories

| Directory    | Purpose |
|-------------|---------|
| `src/kiara/` | Main Python package: model, training, config, tokenizer, attention, and utils |
| `scripts/`   | Executable scripts: train, evaluate, generate, serve |
| `configs/`   | YAML configuration files (e.g. default, small) |
| `data/`      | Training and validation text files (e.g. train.txt, val.txt) |
| `checkpoints/` | Saved model checkpoints (.pt files) |
| `tests/`     | Pytest unit tests |
| `documentation/` | All project documentation (including this file) |

---

## Source Package: `src/kiara/`

### Core Modules

#### `model.py`
- **Purpose:** GPT-style transformer model.
- **Contents:**
  - `LayerNorm`, `GELU`, `FeedForward`: Building blocks used in each transformer block.
  - `TransformerBlock`: One block of multi-head attention + feed-forward with pre-norm and residuals.
  - `GPTModel`: Full model: token + position embeddings, stacked transformer blocks, final layer norm, and LM head that outputs logits over the vocabulary.
- **Usage:** Imported by training and serving code; instantiated with a config dict (`vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`).

#### `attention.py`
- **Purpose:** Self-attention and multi-head attention used by the transformer.
- **Contents:**
  - `SelfAttention`: Single-head attention (Q, K, V projections; scaled dot-product attention; optional causal mask).
  - `MultiHeadAttention`: Multiple `SelfAttention` heads with a final linear projection; used by `TransformerBlock` in `model.py`.
- **Usage:** `model.py` imports `MultiHeadAttention` and uses it inside each `TransformerBlock`.

#### `training.py`
- **Purpose:** Data loading, loss computation, training loop, and simple text generation.
- **Contents:**
  - `GPTDataset`: Builds (input, target) token sequences from raw text using a sliding window (next-token prediction).
  - `create_dataloader`: Builds a PyTorch `DataLoader` from text and tokenizer.
  - `calc_loss_batch` / `calc_loss_loader`: Forward pass and cross-entropy loss for a batch or over a loader.
  - `train_model_simple`: Training loop (forward, loss, backward, optimizer step, optional validation and sample generation).
  - `generate_text_simple` / `generate_text_sampling`: Autoregressive text generation (greedy or sampling).
- **Usage:** Used by `scripts/train.py`; generation helpers are used by `scripts/generate.py` and `scripts/serve.py`.

#### `tokenizer.py`
- **Purpose:** Simple tokenizer utilities (e.g. whitespace/punctuation-based).
- **Contents:** `SimpleTokenizer` with `tokenize`, `encode`, `decode`, and vocabulary handling.
- **Usage:** Scripts typically use **tiktoken** (e.g. GPT-2 encoding) for training/generation; this module is available for custom or minimal tokenization needs.

#### `config.py`
- **Purpose:** Central configuration for model, training, data, and logging.
- **Contents:**
  - Dataclasses: `ModelConfig`, `TrainingConfig`, `DataConfig`, `LoggingConfig`, and top-level `Config`.
  - `Config.from_yaml()`, `Config.from_json()`, `Config.from_dict()` for loading; `save()` for writing YAML/JSON.
  - `load_config_from_env()` to build config from environment variables.
- **Usage:** `scripts/train.py` (and optionally other scripts) load config from YAML or env and pass the resulting `Config` (or a dict derived from it) to the model and training code.

### Utilities: `src/kiara/utils/`

| File           | Purpose |
|----------------|---------|
| `checkpoint.py` | `CheckpointManager`: save/load checkpoints, keep last N, copy best by validation loss to `best_model.pt`. |
| `logging.py`    | Structured logging setup (e.g. level, file, format) used across scripts. |
| `metrics.py`    | Evaluation metrics (e.g. perplexity, accuracy) used during validation/evaluation. |

### CLI: `src/kiara/cli/`
- **Purpose:** Optional command-line entry points (e.g. for `kiara train`, `kiara evaluate`, `kiara generate`) if registered in `setup.py`.
- **Contents:** `train.py`, `evaluate.py`, `generate.py` in `cli/` define CLI logic that can call the same core functions used by the scripts.

---

## Scripts: `scripts/`

These are the main entry points you run from the command line.

| Script        | Purpose |
|---------------|---------|
| `train.py`    | Load config (YAML or env), create model and tokenizer (e.g. tiktoken), build train/val dataloaders from `data/train.txt` and `data/val.txt`, run training via `train_model_simple`, and save checkpoints using `CheckpointManager`. Supports `--config` and `--resume`. |
| `evaluate.py` | Load a checkpoint, build model from checkpoint config, run evaluation on a dataset (e.g. from `data/val.txt`) and report loss/metrics (e.g. perplexity). |
| `generate.py` | Load a checkpoint, build model, run text generation from a prompt (greedy or sampling with temperature/top-k). |
| `serve.py`    | Load a checkpoint, build model, start a FastAPI server that exposes a generation endpoint (e.g. `/generate`) for HTTP requests. |

All scripts that need a model load the checkpoint from disk (e.g. `checkpoints/best_model.pt`) and reconstruct the model from the checkpoint’s stored config; no database is involved.

---

## Configuration: `configs/`

| File            | Purpose |
|-----------------|---------|
| `default.yaml`  | Default model (e.g. ~100M params) and training settings (batch size, epochs, learning rate, eval/save frequency, paths for data, checkpoints, logs). |
| `small.yaml`    | Smaller model and shorter training (e.g. fewer layers/embedding size, smaller batch, fewer steps) for quick experiments. |

Config fields map to the dataclasses in `config.py` (model, training, data, logging, plus top-level options like `checkpoint_dir`, `device`, `seed`).

---

## Tests: `tests/`

| File              | Purpose |
|-------------------|---------|
| `test_model.py`   | Tests for model architecture (shape, forward pass, parameter count, etc.). |
| `test_training.py`| Tests for dataset, dataloader, loss, and training/generation utilities. |
| `test_config.py`  | Tests for config load/save (YAML, JSON) and defaults. |
| `test_utils.py`   | Tests for checkpoint, logging, metrics, etc. |

Run with: `pytest tests/` or `make test`.

---

## Root-Level Files

| File               | Purpose |
|--------------------|---------|
| `setup.py`         | Package installation and optional CLI entry points for the `kiara` package. |
| `pyproject.toml`   | Project metadata, tool config (e.g. black, isort, pytest, mypy). |
| `requirements.txt` | Python dependencies (PyTorch, FastAPI, tiktoken, etc.). |
| `Makefile`         | Shortcuts for install, test, lint, format, train, docker build. |
| `Dockerfile`       | Image definition for running training/serving in Docker. |
| `docker-compose.yml` | Services for training, evaluation, and API server with volumes for data and checkpoints. |
| `.env.example`     | Example environment variables (can override config). |

---

## Data and Checkpoint Locations

- **Training/validation data:** Plain text files, e.g. `data/train.txt`, `data/val.txt` (paths set in config). The model does not store this text; it only learns from it during training.
- **Trained model (“memory”):** Stored in `checkpoints/`:
  - `checkpoints/best_model.pt`: Best model by validation loss (used for eval, generate, serve).
  - `checkpoints/checkpoint_epoch{N}_step{M}.pt`: Periodic checkpoints during training.
- **Logs:** Optional log files under `logs/` (configurable in config).

---

## Flow Summary

1. **Training:** `scripts/train.py` reads config and data → builds model and dataloaders → runs `train_model_simple` → `CheckpointManager` writes `.pt` files to `checkpoints/`.
2. **Evaluation:** `scripts/evaluate.py` loads a checkpoint → builds model → evaluates on validation data → reports metrics.
3. **Generation:** `scripts/generate.py` or `scripts/serve.py` loads a checkpoint → builds model → runs autoregressive generation from a prompt.

All “memory” of the training data is encoded in the model weights saved in `checkpoints/*.pt`; there is no separate database for the book or any other training text.

For more detail on layout and extension points, see [Project Structure](PROJECT_STRUCTURE.md). For setup and usage, see [Quick Start](QUICKSTART.md) and [Configuration Guide](CONFIGURATION.md).
