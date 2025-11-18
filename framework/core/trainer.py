"""
Training module for the framework
Handles model training with support for mixed precision, gradient accumulation, and early stopping
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from ..config.base_config import ExperimentConfig
from ..utils.logger import Logger
from ..utils.checkpoint import CheckpointManager


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def is_better(self, score: float) -> bool:
        """Check if score is better than best score"""
        if self.best_score is None:
            return True

        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score


class Trainer:
    """
    Main training class with support for:
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Checkpointing
    - Multi-GPU training
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device: str = "cuda",
        logger: Optional[Logger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """
        Initialize trainer

        Args:
            config: Experiment configuration
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            logger: Logger instance
            checkpoint_manager: Checkpoint manager
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # Mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None

        # Early stopping
        self.early_stopping = None
        if config.training.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                mode='min'
            )

        # Multi-GPU support
        if config.training.use_multi_gpu and torch.cuda.device_count() > 1:
            if logger:
                logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(model, device_ids=config.training.gpu_ids)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_frames = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config.training.max_epochs}",
            disable=self.logger is None
        )

        for batch_idx, batch in enumerate(pbar):
            # Get data
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)

            # Forward pass with mixed precision
            if self.config.training.mixed_precision:
                with autocast():
                    outputs = self.model(features, mask)
                    loss = self.criterion(outputs, labels, mask)
            else:
                outputs = self.model(features, mask)
                loss = self.criterion(outputs, labels, mask)

            # Gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps

            # Backward pass
            if self.config.training.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Compute metrics
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps

            # Compute accuracy
            predictions = torch.argmax(outputs, dim=1)
            if mask is not None:
                correct = ((predictions == labels) * mask).sum().item()
                total_frames += mask.sum().item()
            else:
                correct = (predictions == labels).sum().item()
                total_frames += labels.numel()
            total_correct += correct

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct / labels.numel():.4f}"
            })

            # Log to TensorBoard
            if self.logger and self.global_step % self.config.visualization.log_interval == 0:
                self.logger.log_scalar('train/batch_loss', loss.item(), self.global_step)

        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_frames if total_frames > 0 else 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_frames = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=self.logger is None):
                # Get data
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(features, mask)
                loss = self.criterion(outputs, labels, mask)

                # Compute metrics
                total_loss += loss.item()

                # Compute accuracy
                predictions = torch.argmax(outputs, dim=1)
                if mask is not None:
                    correct = ((predictions == labels) * mask).sum().item()
                    total_frames += mask.sum().item()
                else:
                    correct = (predictions == labels).sum().item()
                    total_frames += labels.numel()
                total_correct += correct

        # Validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_frames if total_frames > 0 else 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
        }

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Main training loop

        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        if self.logger:
            self.logger.info("Starting training...")
            self.logger.info(f"Total epochs: {self.config.training.max_epochs}")
            self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, self.config.training.max_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])

            if self.logger:
                self.logger.log_metrics(train_metrics, epoch, prefix="train/")

            # Validate
            if epoch % self.config.training.val_interval == 0:
                val_metrics = self.validate()
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])

                if self.logger:
                    self.logger.log_metrics(val_metrics, epoch, prefix="val/")

                # Check if best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_acc = val_metrics['accuracy']
                    if self.logger:
                        self.logger.info(f"New best model! Val Loss: {self.best_val_loss:.4f}, "
                                       f"Val Acc: {self.best_val_acc:.4f}")

                # Save checkpoint
                if self.checkpoint_manager and self.config.training.save_checkpoints:
                    if epoch % self.config.training.checkpoint_interval == 0 or is_best:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            metrics={
                                'train_loss': train_metrics['loss'],
                                'train_acc': train_metrics['accuracy'],
                                'val_loss': val_metrics['loss'],
                                'val_acc': val_metrics['accuracy'],
                            },
                            is_best=is_best,
                        )

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(val_metrics['loss']):
                        if self.logger:
                            self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

        if self.logger:
            self.logger.info("Training completed!")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")

        return history

    def resume_training(self, checkpoint_path: str):
        """
        Resume training from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.checkpoint_manager:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                checkpoint_path,
                self.model,
                self.optimizer,
                self.device
            )
            self.current_epoch = checkpoint_info['epoch']
            if self.logger:
                self.logger.info(f"Resumed training from epoch {self.current_epoch}")
