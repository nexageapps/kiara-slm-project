"""Text generation script for Kiara SLM."""

import argparse
import torch
import tiktoken
from pathlib import Path

from kiara.model import GPTModel
from kiara.training import generate_text_simple, generate_text_sampling
from kiara.utils import setup_logger, CheckpointManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with Kiara SLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    """Main generation function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(name="kiara.generate", log_level="INFO")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model_config = checkpoint.get('config', {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
    })
    
    model = GPTModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Generate text
    logger.info(f"\nPrompt: {args.prompt}")
    logger.info("-" * 60)
    
    encoded = tokenizer.encode(args.prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if args.temperature == 0:
            # Greedy decoding
            token_ids = generate_text_simple(
                model=model,
                idx=encoded_tensor,
                max_new_tokens=args.max_tokens,
                context_size=model_config["context_length"]
            )
        else:
            # Sampling
            token_ids = generate_text_sampling(
                model=model,
                idx=encoded_tensor,
                max_new_tokens=args.max_tokens,
                context_size=model_config["context_length"],
                temperature=args.temperature,
                top_k=args.top_k
            )
    
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(decoded_text)
    print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
