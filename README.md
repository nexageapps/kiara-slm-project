# Small Language Model (SLM) - From Scratch

A project to build a Small Language Model following Sebastian Raschka's "Build a Large Language Model (From Scratch)" approach.

## Project Overview

This project implements a GPT-like language model from scratch, covering:
- Text preprocessing and tokenization
- Attention mechanisms (self-attention, multi-head attention)
- Transformer architecture
- Pre-training and fine-tuning
- Model evaluation

## Project Structure

```
slm-project/
├── data/               # Training data
├── src/
│   ├── tokenizer.py   # Tokenization implementation
│   ├── model.py       # Model architecture
│   ├── attention.py   # Attention mechanisms
│   ├── training.py    # Training loop
│   └── utils.py       # Utility functions
├── notebooks/         # Jupyter notebooks for experimentation
├── tests/            # Unit tests
├── configs/          # Configuration files
└── requirements.txt  # Dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- tiktoken (for tokenization)

### Installation

```bash
pip install -r requirements.txt
```

## Implementation Roadmap

### Phase 1: Foundation (Chapters 1-2)
- [x] Project setup
- [ ] Understand LLM basics
- [ ] Implement text preprocessing
- [ ] Build simple tokenizer (BPE)

### Phase 2: Attention Mechanism (Chapter 3)
- [ ] Implement self-attention
- [ ] Add causal masking
- [ ] Build multi-head attention

### Phase 3: Model Architecture (Chapter 4)
- [ ] Implement GPT architecture
- [ ] Add positional encoding
- [ ] Build transformer blocks

### Phase 4: Pre-training (Chapter 5)
- [ ] Implement training loop
- [ ] Add data loading
- [ ] Calculate loss and metrics

### Phase 5: Fine-tuning (Chapter 6)
- [ ] Implement fine-tuning
- [ ] Add instruction following
- [ ] Implement evaluation

## Key Concepts

### Tokenization
Converting text into numerical tokens that the model can process.

### Self-Attention
Mechanism that allows the model to weigh the importance of different words in context.

### Transformer Architecture
The core architecture using attention mechanisms instead of recurrence.

## Resources

- Book: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
- GitHub: https://github.com/rasbt/LLMs-from-scratch
- PyTorch Documentation: https://pytorch.org/docs/

## Next Steps

1. Set up your development environment
2. Start with Chapter 2: text preprocessing and tokenization
3. Implement the simple tokenizer
4. Move on to attention mechanisms

## License

MIT License
# kiara-slm-project
