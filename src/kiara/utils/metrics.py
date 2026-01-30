"""Metrics calculation utilities."""

import torch
import torch.nn as nn
from typing import Tuple


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()


def calculate_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    predictions = torch.argmax(logits, dim=-1)
    
    if ignore_index is not None:
        mask = targets != ignore_index
        correct = ((predictions == targets) & mask).sum().item()
        total = mask.sum().item()
    else:
        correct = (predictions == targets).sum().item()
        total = targets.numel()
    
    return correct / total if total > 0 else 0.0


def calculate_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        k: Number of top predictions to consider
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    
    if ignore_index is not None:
        mask = targets != ignore_index
        correct = ((top_k_preds == targets_expanded).any(dim=-1) & mask).sum().item()
        total = mask.sum().item()
    else:
        correct = (top_k_preds == targets_expanded).any(dim=-1).sum().item()
        total = targets.numel()
    
    return correct / total if total > 0 else 0.0
