"""
Checkpoint management utilities
Handles saving and loading model checkpoints
"""

import os
import torch
import shutil
from typing import Dict, Optional, Any
from pathlib import Path


class CheckpointManager:
    """
    Manages model checkpoints including saving, loading, and cleanup
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = False,
        max_checkpoints: int = 5,
        logger=None
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: If True, only save the best model
            max_checkpoints: Maximum number of checkpoints to keep
            logger: Logger instance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        self.logger = logger

        self.best_metric = None
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if extra_state is not None:
            checkpoint.update(extra_state)

        # Save regular checkpoint
        if not self.save_best_only:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append(checkpoint_path)

            if self.logger:
                self.logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Clean up old checkpoints
            self._cleanup_checkpoints()

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            if self.logger:
                self.logger.info(f"Saved best model: {best_path}")
            return str(best_path)

        return str(checkpoint_path) if not self.save_best_only else ""

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            device: Device to load checkpoint to

        Returns:
            Dictionary containing epoch and metrics
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.logger:
            self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            self.logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
        }

    def load_best_model(
        self,
        model: torch.nn.Module,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load the best model checkpoint

        Args:
            model: Model to load weights into
            device: Device to load checkpoint to

        Returns:
            Dictionary containing epoch and metrics
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        return self.load_checkpoint(str(best_path), model, device=device)

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None

        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoints[-1])

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to keep only max_checkpoints"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by modification time
            self.checkpoints.sort(key=lambda x: os.path.getmtime(x))

            # Remove oldest checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    if self.logger:
                        self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        return list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best model checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pth"
        return str(best_path) if best_path.exists() else None
