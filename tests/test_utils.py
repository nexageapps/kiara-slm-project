"""Tests for utility functions."""

import pytest
import torch
from kiara.utils.metrics import calculate_perplexity, calculate_accuracy


def test_calculate_perplexity():
    """Test perplexity calculation."""
    loss = 2.0
    perplexity = calculate_perplexity(loss)
    
    assert perplexity > 0
    assert abs(perplexity - 7.389) < 0.01  # e^2 ≈ 7.389


def test_calculate_accuracy():
    """Test accuracy calculation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    # Create dummy logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    accuracy = calculate_accuracy(logits, targets)
    
    assert 0 <= accuracy <= 1


def test_calculate_accuracy_perfect():
    """Test accuracy with perfect predictions."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create logits where correct token has highest score
    logits = torch.randn(batch_size, seq_len, vocab_size)
    for i in range(batch_size):
        for j in range(seq_len):
            logits[i, j, targets[i, j]] = 100.0  # Very high score
    
    accuracy = calculate_accuracy(logits, targets)
    
    assert accuracy == 1.0
