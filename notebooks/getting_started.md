# Getting Started with Building Your SLM

This notebook will guide you through the basics of building a Small Language Model following Sebastian Raschka's book.

## Step 1: Understanding the Components

A language model consists of several key components:

1. **Tokenizer**: Converts text to numbers
2. **Embeddings**: Represents tokens as vectors
3. **Attention**: Allows tokens to attend to each other
4. **Transformer Blocks**: The core architecture
5. **Training Loop**: How the model learns

## Step 2: Install Dependencies

```python
# Run this in your terminal or notebook
# pip install torch numpy tiktoken matplotlib tqdm
```

## Step 3: Test the Tokenizer

```python
import sys
sys.path.append('../src')

from tokenizer import SimpleTokenizer

# Create and test tokenizer
text = "Hello, world! This is a test."
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(text)

print("Vocabulary:", tokenizer.vocab)
print("Encoded:", tokenizer.encode("Hello, world!"))
```

## Step 4: Understanding Attention

Attention is the key mechanism that makes transformers work. It allows each token to "look at" other tokens in the sequence.

```python
import torch
from attention import SelfAttention, MultiHeadAttention

# Create sample input (batch_size=1, seq_len=4, embedding_dim=8)
x = torch.randn(1, 4, 8)

# Test self-attention
attn = SelfAttention(d_in=8, d_out=8)
output = attn(x)
print(f"Self-attention output shape: {output.shape}")

# Test multi-head attention
mh_attn = MultiHeadAttention(d_in=8, d_out=8, context_length=10, num_heads=4)
output = mh_attn(x)
print(f"Multi-head attention output shape: {output.shape}")
```

## Step 5: Create a Small Model

```python
from model import GPTModel, get_gpt_config

# Get configuration for a small model
config = get_gpt_config("small")
config["vocab_size"] = 1000  # Smaller vocab for testing
config["context_length"] = 128  # Shorter context
config["n_layers"] = 4  # Fewer layers

# Create model
model = GPTModel(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Test forward pass
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

with torch.no_grad():
    logits = model(input_ids)
    
print(f"Input shape: {input_ids.shape}")
print(f"Output logits shape: {logits.shape}")
```

## Step 6: Prepare Training Data

For real training, you'll need text data. You can start with:
- Small text files (books, articles)
- Wikipedia dumps
- Project Gutenberg books
- OpenWebText dataset

```python
# Example with simple text
sample_text = """
Your training text goes here.
This could be from books, articles, or any text source.
The model will learn to predict the next token.
"""

# You can use tiktoken for better tokenization
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(sample_text)
print(f"Number of tokens: {len(encoded_text)}")
```

## Next Steps

1. **Read Chapter 2** of Raschka's book: Tokenization in detail
2. **Implement BPE tokenizer**: Better than the simple tokenizer
3. **Read Chapter 3**: Deep dive into attention mechanisms
4. **Read Chapter 4**: Complete GPT architecture
5. **Read Chapter 5**: Pre-training your model
6. **Collect training data**: Find a text corpus
7. **Start training**: Use the training utilities

## Resources

- Book: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
- GitHub: https://github.com/rasbt/LLMs-from-scratch
- Datasets: Hugging Face Datasets (https://huggingface.co/datasets)

## Common Pitfalls to Avoid

1. **Too large models**: Start small (few layers, small embedding dimension)
2. **Not enough data**: You need substantial text for good results
3. **Wrong learning rate**: Start with 3e-4 and adjust
4. **No validation set**: Always split your data
5. **Not monitoring loss**: Track training and validation loss

## Tips for Success

- Start with toy examples to understand each component
- Gradually increase model size as you understand more
- Use pre-trained tokenizers (like tiktoken) initially
- Monitor GPU memory if using CUDA
- Save checkpoints frequently during training
- Experiment with hyperparameters systematically
