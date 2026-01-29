"""
GPT Model Architecture
Based on Sebastian Raschka's LLM book - Chapter 4
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block with multi-head attention and feed-forward network.
    Uses pre-normalization (LayerNorm before attention/FFN).
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Multi-head attention
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"]
        )
        
        # Feed-forward network
        self.ff = FeedForward(cfg)
        
        # Layer normalization
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # Dropout
        self.drop_resid = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        # Feed-forward with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        return x


class GPTModel(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model.
    
    Architecture:
    1. Token embeddings + positional embeddings
    2. Multiple transformer blocks
    3. Layer normalization
    4. Linear output layer (language modeling head)
    """
    
    def __init__(self, cfg):
        """
        Initialize GPT model.
        
        Args:
            cfg: Configuration dictionary with keys:
                - vocab_size: Size of vocabulary
                - emb_dim: Embedding dimension
                - context_length: Maximum sequence length
                - n_heads: Number of attention heads
                - n_layers: Number of transformer blocks
                - drop_rate: Dropout rate
        """
        super().__init__()
        
        # Token embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # Positional embeddings
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Embedding dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Final layer normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # Output projection to vocabulary
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        tok_embeds = self.tok_emb(idx)  # (batch_size, seq_len, emb_dim)
        
        # Positional embeddings
        pos_indices = torch.arange(seq_len, device=idx.device)
        pos_embeds = self.pos_emb(pos_indices)  # (seq_len, emb_dim)
        
        # Combine embeddings
        x = tok_embeds + pos_embeds  # Broadcasting: (batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)
        
        # Pass through transformer blocks
        x = self.trf_blocks(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.out_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits


def get_gpt_config(model_size: str = "small") -> dict:
    """
    Get configuration for different GPT model sizes.
    
    Args:
        model_size: One of "small", "medium", "large"
        
    Returns:
        Configuration dictionary
    """
    configs = {
        "small": {
            "vocab_size": 50257,      # GPT-2 vocab size
            "context_length": 256,     # Shorter for faster training
            "emb_dim": 768,           # Embedding dimension
            "n_heads": 12,            # Number of attention heads
            "n_layers": 12,           # Number of transformer blocks
            "drop_rate": 0.1,         # Dropout rate
        },
        "medium": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1024,
            "n_heads": 16,
            "n_layers": 24,
            "drop_rate": 0.1,
        },
        "large": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1280,
            "n_heads": 20,
            "n_layers": 36,
            "drop_rate": 0.1,
        }
    }
    
    return configs.get(model_size, configs["small"])


if __name__ == "__main__":
    # Example usage
    cfg = get_gpt_config("small")
    cfg["context_length"] = 256  # Smaller for testing
    
    print("Model Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = GPTModel(cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, cfg["vocab_size"], (batch_size, seq_len))
    
    print(f"\nInput shape: {idx.shape}")
    
    with torch.no_grad():
        logits = model(idx)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: (batch_size={batch_size}, seq_len={seq_len}, vocab_size={cfg['vocab_size']})")
