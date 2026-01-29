"""
Simple Tokenizer Implementation
Based on Sebastian Raschka's LLM book - Chapter 2
"""

import re
from typing import List, Dict


class SimpleTokenizer:
    """
    A simple tokenizer that splits text into tokens based on whitespace and punctuation.
    This is a starting point before implementing BPE (Byte Pair Encoding).
    """
    
    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize the tokenizer.
        
        Args:
            vocab: Optional pre-built vocabulary mapping tokens to IDs
        """
        self.vocab = vocab if vocab is not None else {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()} if vocab else {}
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by normalizing whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens using simple regex pattern.
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
        """
        text = self.preprocess_text(text)
        
        # Split on whitespace and punctuation, keeping punctuation as separate tokens
        # This pattern keeps words together and separates punctuation
        pattern = r'([,.:;?_!"()\']|--|\s)'
        tokens = re.split(pattern, text)
        
        # Remove empty strings and whitespace-only tokens
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocab(self, text: str) -> None:
        """
        Build vocabulary from text.
        
        Args:
            text: Input text to build vocabulary from
        """
        tokens = self.tokenize(text)
        unique_tokens = sorted(set(tokens))
        
        # Add special tokens
        self.vocab = {"<|endoftext|>": 0}
        
        # Add unique tokens
        for i, token in enumerate(unique_tokens, start=1):
            self.vocab[token] = i
            
        # Build inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs, using vocab or adding unknown tokens
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Handle unknown tokens - you might want to add <UNK> token
                print(f"Warning: Unknown token '{token}' encountered")
                
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        tokens = [self.inverse_vocab.get(id_, "<UNK>") for id_ in token_ids]
        
        # Join tokens with space, but handle punctuation specially
        text = " ".join(tokens)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.:;?!"])', r'\1', text)
        
        return text
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.vocab)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Hello, world! This is a simple tokenizer.
    It splits text into tokens. How cool is that?
    """
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(sample_text)
    
    print("Vocabulary size:", len(tokenizer))
    print("\nVocabulary:", tokenizer.vocab)
    
    # Encode text
    encoded = tokenizer.encode("Hello, world!")
    print("\nEncoded:", encoded)
    
    # Decode back
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
