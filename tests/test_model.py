"""Tests for model architecture."""

import pytest
import torch
from kiara.model import GPTModel, get_gpt_config


def test_gpt_model_creation():
    """Test GPT model can be created."""
    config = get_gpt_config("small")
    config["context_length"] = 64  # Smaller for testing
    
    model = GPTModel(config)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_gpt_model_forward():
    """Test GPT model forward pass."""
    config = get_gpt_config("small")
    config["context_length"] = 64
    
    model = GPTModel(config)
    
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(idx)
    
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])


def test_gpt_model_parameter_count():
    """Test model has expected number of parameters."""
    config = {
        "vocab_size": 50257,
        "context_length": 64,
        "emb_dim": 256,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
    }
    
    model = GPTModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Should have millions of parameters
    assert total_params > 1_000_000


def test_model_configs():
    """Test different model size configurations."""
    for size in ["small", "medium", "large"]:
        config = get_gpt_config(size)
        assert "vocab_size" in config
        assert "emb_dim" in config
        assert "n_heads" in config
        assert "n_layers" in config
