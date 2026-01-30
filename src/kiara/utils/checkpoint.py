"""Checkpoint management utilities."""

import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from kiara.utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manage model checkpoints with automatic cleanup."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 3,
        save_best: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best checkpoint separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric = float('inf')
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            config: Model configuration
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint if applicable
        if self.save_best and 'val_loss' in metrics:
            if metrics['val_loss'] < self.best_metric:
                self.best_metric = metrics['val_loss']
                best_path = self.checkpoint_dir / "best_model.pt"
                shutil.copy(checkpoint_path, best_path)
                logger.info(f"New best model saved with val_loss: {self.best_metric:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Step: {checkpoint.get('step', 'N/A')}")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {best_path}")
        return self.load_checkpoint(best_path, model, optimizer, device)
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Keep only the last N checkpoints
        for checkpoint in checkpoints[self.keep_last_n:]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        return sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
