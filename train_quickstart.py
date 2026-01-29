"""
Quick Start Training Script
A minimal example to get started training your SLM

This script demonstrates:
1. Creating a small GPT model
2. Preparing simple training data
3. Running a training loop
4. Generating text from the trained model
"""

import torch
import torch.nn as nn
from src.model import GPTModel
from src.training import create_dataloader, train_model_simple, generate_text_simple
import tiktoken


def main():
    # Configuration
    print("=" * 60)
    print("SLM Quick Start Training")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Model configuration (very small for quick testing)
    GPT_CONFIG = {
        "vocab_size": 50257,      # GPT-2 tokenizer vocab size
        "context_length": 256,    # Shorter context for faster training
        "emb_dim": 256,          # Smaller embedding dimension
        "n_heads": 4,            # Fewer attention heads
        "n_layers": 4,           # Fewer transformer blocks
        "drop_rate": 0.1,
    }
    
    print("\nModel Configuration:")
    for key, value in GPT_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = GPTModel(GPT_CONFIG)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Prepare tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Sample training text (replace with your own dataset)
    # For real training, load a larger text file
    with open("sample_data.txt", "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"\nTraining data size: {len(text_data)} characters")
    
    # Split into train and validation
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    # Create data loaders
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        shuffle=True,
        drop_last=True
    )
    
    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        shuffle=False,
        drop_last=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.1
    )
    
    # Training parameters
    num_epochs = 10
    eval_freq = 50
    eval_iter = 5
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Train model
    training_history = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        start_context="The future of AI",
        tokenizer=tokenizer
    )
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': GPT_CONFIG,
        'training_history': training_history
    }, 'trained_model.pth')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel saved to: trained_model.pth")
    
    # Generate some text
    print("\n" + "=" * 60)
    print("Generating Sample Text")
    print("=" * 60)
    
    model.eval()
    
    prompts = [
        "Once upon a time",
        "The future of AI",
        "In a world where"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        encoded = tokenizer.encode(prompt)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model,
                idx=encoded_tensor,
                max_new_tokens=50,
                context_size=GPT_CONFIG["context_length"]
            )
        
        decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
        print(decoded_text)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Check if sample data exists, if not create it
    import os
    if not os.path.exists("sample_data.txt"):
        print("Creating sample_data.txt...")
        sample_text = """The development of artificial intelligence has been one of the most significant technological advances of the 21st century.
Machine learning algorithms have revolutionized how we process information and make decisions.
Deep learning, a subset of machine learning, uses neural networks with multiple layers to learn from data.
Natural language processing enables computers to understand and generate human language.
Computer vision allows machines to interpret and understand visual information from the world.
Reinforcement learning helps AI systems learn through trial and error, similar to how humans learn.
The transformer architecture, introduced in 2017, became the foundation for modern language models.
Large language models like GPT demonstrate impressive capabilities in text generation and understanding.
Ethical considerations in AI development include fairness, transparency, and accountability.
The future of AI holds promise for solving complex problems in healthcare, climate, and education.
"""
        with open("sample_data.txt", "w", encoding="utf-8") as f:
            f.write(sample_text * 20)  # Repeat to have more training data
        print("Sample data created!\n")
    
    main()
