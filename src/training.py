"""
Training Utilities
Based on Sebastian Raschka's LLM book - Chapter 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class GPTDataset(Dataset):
    """
    Dataset for GPT training.
    Creates input-target pairs for next-token prediction.
    """
    
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """
        Initialize dataset.
        
        Args:
            txt: Input text string
            tokenizer: Tokenizer with encode() method
            max_length: Maximum sequence length (context_length)
            stride: Step size for sliding window
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        
        # Encode the entire text
        token_ids = tokenizer.encode(txt)
        
        # Create input-target pairs using sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt: str, tokenizer, batch_size: int = 4, 
                      max_length: int = 256, stride: int = 128, 
                      shuffle: bool = True, drop_last: bool = True,
                      num_workers: int = 0):
    """
    Create DataLoader for GPT training.
    
    Args:
        txt: Input text
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for sliding window
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, 
                    model: nn.Module, device: torch.device) -> torch.Tensor:
    """
    Calculate loss for a single batch.
    
    Args:
        input_batch: Input token IDs of shape (batch_size, seq_len)
        target_batch: Target token IDs of shape (batch_size, seq_len)
        model: GPT model
        device: Device to run on
        
    Returns:
        Loss tensor
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Forward pass
    logits = model(input_batch)
    
    # Reshape for cross-entropy loss
    # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    # targets: (batch_size, seq_len) -> (batch_size * seq_len)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        target_batch.flatten()
    )
    
    return loss


def calc_loss_loader(data_loader: DataLoader, model: nn.Module, 
                     device: torch.device, num_batches: int = None) -> float:
    """
    Calculate average loss over entire dataloader.
    
    Args:
        data_loader: DataLoader instance
        model: GPT model
        device: Device to run on
        num_batches: Optional limit on number of batches to evaluate
        
    Returns:
        Average loss
    """
    total_loss = 0.0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
            
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    return total_loss / num_batches


def train_model_simple(model: nn.Module, train_loader: DataLoader, 
                       val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                       device: torch.device, num_epochs: int,
                       eval_freq: int, eval_iter: int, 
                       start_context: str = None, tokenizer = None) -> dict:
    """
    Simple training loop for GPT model.
    
    Args:
        model: GPT model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        device: Device to train on
        num_epochs: Number of epochs
        eval_freq: Evaluate every N steps
        eval_iter: Number of batches for evaluation
        start_context: Optional text to generate from during training
        tokenizer: Tokenizer for text generation
        
    Returns:
        Dictionary with training history
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # Generate sample text at end of epoch
        if start_context is not None and tokenizer is not None:
            model.eval()
            context_size = model.pos_emb.weight.shape[0]
            
            encoded = tokenizer.encode(start_context)
            encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
            
            print(f"\n{'='*50}\n{start_context}", end="")
            
            with torch.no_grad():
                token_ids = generate_text_simple(
                    model=model,
                    idx=encoded_tensor,
                    max_new_tokens=50,
                    context_size=context_size
                )
                
                decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
                print(decoded_text.replace(start_context, ""))
                print(f"\n{'='*50}\n")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": track_tokens_seen
    }


def generate_text_simple(model: nn.Module, idx: torch.Tensor, 
                         max_new_tokens: int, context_size: int) -> torch.Tensor:
    """
    Generate text using the trained model.
    
    Args:
        model: Trained GPT model
        idx: Initial context tokens of shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context length
        
    Returns:
        Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # Crop context if it exceeds maximum size
        idx_cond = idx[:, -context_size:]
        
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on last time step
        logits = logits[:, -1, :]
        
        # Get most likely token (greedy decoding)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def generate_text_sampling(model: nn.Module, idx: torch.Tensor, 
                          max_new_tokens: int, context_size: int,
                          temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
    """
    Generate text with temperature sampling and optional top-k filtering.
    
    Args:
        model: Trained GPT model
        idx: Initial context tokens
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context length
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        
    Returns:
        Generated token IDs
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.full_like(logits, float('-inf')),
                    logits
                )
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


if __name__ == "__main__":
    print("Training utilities loaded successfully!")
    print("\nKey functions:")
    print("  - GPTDataset: Dataset for next-token prediction")
    print("  - create_dataloader: Create DataLoader")
    print("  - train_model_simple: Training loop")
    print("  - generate_text_simple: Greedy text generation")
    print("  - generate_text_sampling: Sampling-based generation")
