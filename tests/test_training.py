"""Tests for training utilities."""

import pytest
import torch
import tiktoken
from kiara.training import GPTDataset, create_dataloader, calc_loss_batch
from kiara.model import GPTModel


def test_gpt_dataset():
    """Test GPT dataset creation."""
    text = "This is a test sentence. " * 100
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = GPTDataset(text, tokenizer, max_length=32, stride=16)
    
    assert len(dataset) > 0
    
    input_ids, target_ids = dataset[0]
    assert input_ids.shape == target_ids.shape
    assert len(input_ids) == 32


def test_create_dataloader():
    """Test dataloader creation."""
    text = "This is a test sentence. " * 100
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataloader = create_dataloader(
        text,
        tokenizer,
        batch_size=4,
        max_length=32,
        stride=16,
        shuffle=True,
        drop_last=True
    )
    
    assert len(dataloader) > 0
    
    for input_batch, target_batch in dataloader:
        assert input_batch.shape[0] == 4  # batch size
        assert input_batch.shape[1] == 32  # sequence length
        assert input_batch.shape == target_batch.shape
        break


def test_calc_loss_batch():
    """Test loss calculation."""
    config = {
        "vocab_size": 50257,
        "context_length": 32,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
    }
    
    model = GPTModel(config)
    device = torch.device("cpu")
    
    batch_size = 2
    seq_len = 32
    input_batch = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    target_batch = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
