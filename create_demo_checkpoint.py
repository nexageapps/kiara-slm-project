"""Create a demo checkpoint for HF Spaces."""
import torch
from src.kiara.model import GPTModel

# Model config matching small.yaml
config = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
}

# Create model
model = GPTModel(config)

# Create checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'epoch': 0,
    'global_step': 0,
}

# Save
import os
os.makedirs('checkpoints', exist_ok=True)
torch.save(checkpoint, 'checkpoints/best_model.pt')
print("✅ Created demo checkpoint at checkpoints/best_model.pt")
print("⚠️  Note: This is an untrained model - it will generate random text")
