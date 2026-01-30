"""Evaluation script for Kiara SLM."""

import argparse
import torch
import tiktoken
from pathlib import Path
from tqdm import tqdm

from kiara.model import GPTModel
from kiara.training import create_dataloader, calc_loss_loader
from kiara.utils import setup_logger, calculate_perplexity, calculate_accuracy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Kiara SLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(name="kiara.evaluate", log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("Kiara SLM Evaluation")
    logger.info("=" * 60)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"\nLoading checkpoint from {args.checkpoint}")
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
    
    logger.info("Model loaded successfully")
    
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load evaluation data
    logger.info(f"\nLoading evaluation data from {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        eval_data = f.read()
    
    logger.info(f"Evaluation data size: {len(eval_data)} characters")
    
    # Create data loader
    eval_loader = create_dataloader(
        eval_data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    logger.info(f"Evaluation batches: {len(eval_loader)}")
    
    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("Running Evaluation")
    logger.info("=" * 60)
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_batch, target_batch in tqdm(eval_loader, desc="Evaluating"):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            logits = model(input_batch)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            
            # Calculate accuracy
            accuracy = calculate_accuracy(logits, target_batch)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.2f}")
    logger.info(f"Token Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
