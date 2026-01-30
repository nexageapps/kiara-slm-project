"""Production training script for Kiara SLM."""

import argparse
import torch
import tiktoken
from pathlib import Path
from dotenv import load_dotenv

from kiara.model import GPTModel
from kiara.training import create_dataloader, train_model_simple
from kiara.config import Config, load_config_from_env
from kiara.utils import setup_logger, CheckpointManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Kiara SLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
    else:
        config = load_config_from_env()
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    # Setup logging
    logger = setup_logger(
        name="kiara.train",
        log_level=config.logging.log_level,
        log_dir=config.logging.log_dir,
        log_file="train.log"
    )
    
    logger.info("=" * 60)
    logger.info("Kiara SLM Training")
    logger.info("=" * 60)
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log configuration
    logger.info("\nModel Configuration:")
    for key, value in config.model.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Create model
    model_config = {
        "vocab_size": config.model.vocab_size,
        "context_length": config.model.context_length,
        "emb_dim": config.model.emb_dim,
        "n_heads": config.model.n_heads,
        "n_layers": config.model.n_layers,
        "drop_rate": config.model.drop_rate,
    }
    
    model = GPTModel(model_config)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nTotal parameters: {total_params:,}")
    
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load data
    logger.info("\nLoading training data...")
    
    if config.data.train_data_path and Path(config.data.train_data_path).exists():
        with open(config.data.train_data_path, "r", encoding="utf-8") as f:
            train_data = f.read()
    else:
        raise FileNotFoundError(f"Training data not found at {config.data.train_data_path}")
    
    if config.data.val_data_path and Path(config.data.val_data_path).exists():
        with open(config.data.val_data_path, "r", encoding="utf-8") as f:
            val_data = f.read()
    else:
        # Split training data if validation data not provided
        split_idx = int(0.9 * len(train_data))
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    
    logger.info(f"Training data size: {len(train_data)} characters")
    logger.info(f"Validation data size: {len(val_data)} characters")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.model.context_length,
        stride=config.data.stride,
        shuffle=True,
        drop_last=True,
        num_workers=config.data.num_workers
    )
    
    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.model.context_length,
        stride=config.data.stride,
        shuffle=False,
        drop_last=False,
        num_workers=config.data.num_workers
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        keep_last_n=config.training.keep_last_n_checkpoints,
        save_best=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = checkpoint_manager.load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            device=str(device)
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Training
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    training_history = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.training.num_epochs,
        eval_freq=config.training.eval_freq,
        eval_iter=config.training.eval_iter,
        start_context="The future of AI",
        tokenizer=tokenizer
    )
    
    # Save final checkpoint
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.training.num_epochs,
        step=-1,
        metrics={
            'train_loss': training_history['train_losses'][-1],
            'val_loss': training_history['val_losses'][-1]
        },
        config=model_config
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
