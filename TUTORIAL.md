# Complete Tutorial: Building Your SLM Step by Step

This tutorial follows Sebastian Raschka's "Build a Large Language Model (From Scratch)" book.

## Table of Contents
1. [Introduction to LLMs](#introduction)
2. [Chapter 2: Tokenization](#chapter-2)
3. [Chapter 3: Attention Mechanisms](#chapter-3)
4. [Chapter 4: GPT Architecture](#chapter-4)
5. [Chapter 5: Pre-training](#chapter-5)
6. [Chapter 6: Fine-tuning](#chapter-6)

---

## Introduction

Language models predict the next token in a sequence. By training on massive amounts of text, they learn patterns in language and can generate coherent text.

**Key Concepts:**
- **Token**: A unit of text (word, subword, or character)
- **Context Window**: How much text the model can "see" at once
- **Embedding**: Vector representation of a token
- **Attention**: Mechanism for tokens to interact with each other

---

## Chapter 2: Tokenization

Tokenization converts text into numbers that the model can process.

### Why Tokenization Matters

Language models work with numbers, not text. Tokenization is the bridge between human-readable text and model-processable numbers.

### Types of Tokenizers

1. **Character-level**: Each character is a token
   - Pros: Small vocabulary
   - Cons: Long sequences, poor semantic meaning

2. **Word-level**: Each word is a token
   - Pros: Captures word meanings
   - Cons: Huge vocabulary, can't handle unknown words

3. **Subword-level (BPE)**: Balance between characters and words
   - Pros: Handles unknown words, reasonable vocabulary size
   - Cons: More complex to implement

### Your Task

```python
# Start with the simple tokenizer in src/tokenizer.py
from src.tokenizer import SimpleTokenizer

text = "Hello, world! How are you?"
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(text)

# Encode and decode
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
```

**Next Step**: Implement BPE (Byte Pair Encoding) or use tiktoken for GPT-2 style tokenization.

---

## Chapter 3: Attention Mechanisms

Attention is the key innovation that makes transformers work.

### Self-Attention Intuition

Imagine reading this sentence: "The cat sat on the mat because it was tired."

When you read "it," you automatically know it refers to "cat," not "mat." Self-attention allows the model to make these connections.

### How Self-Attention Works

1. **Query, Key, Value**: Each token creates three vectors
   - Query: "What am I looking for?"
   - Key: "What do I contain?"
   - Value: "What information do I have?"

2. **Attention Scores**: Compare queries with all keys
   - High score = strong relationship

3. **Weighted Sum**: Combine values based on attention scores

### Mathematical Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Code Example

```python
from src.attention import SelfAttention

# Create attention layer
attn = SelfAttention(d_in=512, d_out=512)

# Input: (batch_size, sequence_length, embedding_dim)
x = torch.randn(2, 10, 512)

# Output: Same shape
output = attn(x)
```

### Multi-Head Attention

Instead of one attention mechanism, use multiple "heads" in parallel. Each head can learn different relationships:
- Head 1: Subject-verb relationships
- Head 2: Noun-adjective relationships
- Head 3: Long-range dependencies

---

## Chapter 4: GPT Architecture

GPT (Generative Pre-trained Transformer) is built from several components stacked together.

### Architecture Overview

```
Input Text
    ↓
Token Embeddings + Positional Embeddings
    ↓
Transformer Block 1
    ├── Multi-Head Attention
    ├── Add & Normalize
    ├── Feed-Forward Network
    └── Add & Normalize
    ↓
Transformer Block 2
    ↓
    ...
    ↓
Transformer Block N
    ↓
Layer Normalization
    ↓
Output Projection (to Vocabulary)
    ↓
Next Token Probabilities
```

### Key Components

**1. Embeddings**
```python
# Token embeddings: Learn representation for each token
tok_emb = nn.Embedding(vocab_size, emb_dim)

# Positional embeddings: Encode position information
pos_emb = nn.Embedding(max_length, emb_dim)
```

**2. Transformer Block**
```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.attention(self.norm1(x))
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x
```

**3. Causal Masking**

Language models can only look at past tokens (autoregressive). Causal masking prevents looking at future tokens during training.

### Your Task

```python
from src.model import GPTModel, get_gpt_config

# Create a small model
config = get_gpt_config("small")
model = GPTModel(config)

# Test it
input_ids = torch.randint(0, config["vocab_size"], (1, 10))
logits = model(input_ids)  # (1, 10, vocab_size)
```

---

## Chapter 5: Pre-training

Pre-training is where the model learns language patterns from large amounts of text.

### The Training Objective

**Next Token Prediction**: Given a sequence of tokens, predict the next one.

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
```

At each position, the model learns to predict the next token.

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        logits = model(batch.input_ids)
        
        # Calculate loss
        loss = cross_entropy(logits, batch.targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Hyperparameters

Key hyperparameters to tune:

1. **Learning Rate**: Start with 3e-4 for Adam/AdamW
2. **Batch Size**: As large as your GPU memory allows
3. **Context Length**: Trade-off between memory and capability
4. **Number of Layers**: More layers = more capacity but slower
5. **Embedding Dimension**: Larger = more expressive

### Data Requirements

For good results, you need substantial text:
- **Minimum**: Few million tokens (~few books)
- **Good**: 100M+ tokens
- **GPT-3 scale**: 300B+ tokens

### Your Task

```python
# Run the quick start script
python train_quickstart.py
```

---

## Chapter 6: Fine-tuning

After pre-training, fine-tune the model for specific tasks.

### Fine-tuning vs Pre-training

- **Pre-training**: General language understanding (expensive, once)
- **Fine-tuning**: Task-specific adaptation (cheaper, many times)

### Types of Fine-tuning

**1. Supervised Fine-tuning (SFT)**
- Train on high-quality examples
- Format: Instruction → Response

**2. Reinforcement Learning from Human Feedback (RLHF)**
- Use human preferences to improve outputs
- More advanced, beyond basic fine-tuning

### Example: Instruction Following

```python
# Training data format
examples = [
    {
        "instruction": "Summarize this text:",
        "input": "Long article text...",
        "output": "Brief summary..."
    },
    # More examples...
]
```

### Your Task

1. Collect task-specific data
2. Format as instruction-response pairs
3. Fine-tune your pre-trained model
4. Evaluate on held-out test set

---

## Best Practices

### 1. Start Small
- Don't try to build GPT-4 immediately
- Start with tiny models to understand concepts
- Gradually scale up

### 2. Monitor Everything
- Track training loss and validation loss
- Watch for overfitting (train loss << val loss)
- Save checkpoints regularly

### 3. Data Quality Matters
- More data ≠ better model
- Clean, diverse, high-quality text is best
- Remove duplicates and low-quality text

### 4. Use Existing Tools
- Hugging Face Transformers for reference
- tiktoken for tokenization
- Weights & Biases for experiment tracking

### 5. GPU Management
- Use mixed precision training (torch.cuda.amp)
- Gradient accumulation for larger effective batch sizes
- Clear cache between experiments

---

## Debugging Tips

### Model Not Learning?
- Check loss is decreasing
- Verify data loading (print batch samples)
- Try smaller learning rate
- Check for NaN values

### Out of Memory?
- Reduce batch size
- Reduce context length
- Reduce model size (layers, embedding dim)
- Use gradient checkpointing

### Generating Nonsense?
- Train longer (model is underfitted)
- Check tokenizer encoding/decoding
- Verify causal masking is working
- Try different sampling strategies (temperature, top-k)

---

## Next Steps

1. **Implement each component**: Don't skip ahead
2. **Run experiments**: Try different hyperparameters
3. **Read the book**: This guide complements, doesn't replace it
4. **Join communities**: Reddit r/MachineLearning, Discord servers
5. **Build projects**: Apply to real problems

## Resources

- **Book**: "Build a Large Language Model (From Scratch)" - Sebastian Raschka
- **Code**: https://github.com/rasbt/LLMs-from-scratch
- **Papers**: "Attention Is All You Need", "GPT-2", "GPT-3"
- **Courses**: Fast.ai, Stanford CS224N, DeepLearning.AI
- **Tools**: Hugging Face, PyTorch, Weights & Biases

---

Good luck building your language model! 🚀
