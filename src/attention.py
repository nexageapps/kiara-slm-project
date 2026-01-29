"""
Attention Mechanisms
Based on Sebastian Raschka's LLM book - Chapter 3
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Self-attention mechanism that allows tokens to attend to each other.
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        Initialize self-attention.
        
        Args:
            d_in: Input dimension (embedding dimension)
            d_out: Output dimension
        """
        super().__init__()
        
        # Weight matrices for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        # Compute queries, keys, and values
        queries = self.W_query(x)  # (batch_size, seq_len, d_out)
        keys = self.W_key(x)       # (batch_size, seq_len, d_out)
        values = self.W_value(x)   # (batch_size, seq_len, d_out)
        
        # Compute attention scores
        # (batch_size, seq_len, d_out) @ (batch_size, d_out, seq_len)
        # -> (batch_size, seq_len, seq_len)
        attn_scores = queries @ keys.transpose(-2, -1)
        
        # Scale by square root of dimension (for numerical stability)
        d_k = keys.shape[-1]
        attn_scores = attn_scores / math.sqrt(d_k)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute weighted sum of values
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, d_out)
        # -> (batch_size, seq_len, d_out)
        context_vec = attn_weights @ values
        
        return context_vec


class CausalAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive language modeling.
    Prevents tokens from attending to future tokens.
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float = 0.0):
        """
        Initialize causal attention.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
            context_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as a buffer (not a parameter)
        # Upper triangular matrix with 1s above diagonal
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer('mask', mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, d_in = x.shape
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(keys.shape[-1])
        
        # Apply causal mask (set future positions to -inf)
        attn_scores = attn_scores.masked_fill(
            self.mask[:seq_len, :seq_len].bool(), float('-inf')
        )
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vector
        context_vec = attn_weights @ values
        
        return context_vec


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    Runs multiple attention heads in parallel and concatenates results.
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 num_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension (should be divisible by num_heads)
            context_length: Maximum sequence length
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # Single matrices for all heads (more efficient than separate heads)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer('mask', mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, d_in = x.shape
        
        # Project to queries, keys, values
        queries = self.W_query(x)  # (batch_size, seq_len, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_out) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.mask[:seq_len, :seq_len].bool(), float('-inf')
        )
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vec = attn_weights @ values
        
        # Reshape back
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, d_out)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)
        
        # Final linear projection
        output = self.out_proj(context_vec)
        
        return output


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 4
    d_in = 8
    d_out = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_in)
    
    print("Testing Self-Attention:")
    self_attn = SelfAttention(d_in, d_out)
    output = self_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")
    
    print("Testing Causal Attention:")
    causal_attn = CausalAttention(d_in, d_out, context_length=10)
    output = causal_attn(x)
    print(f"Output shape: {output.shape}\n")
    
    print("Testing Multi-Head Attention:")
    mh_attn = MultiHeadAttention(d_in, d_out, context_length=10, num_heads=4)
    output = mh_attn(x)
    print(f"Output shape: {output.shape}")
