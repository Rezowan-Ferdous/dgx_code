"""
Image Classification Task

Unified interface for training, testing, and evaluating classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Literal, List
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from .models import get_classification_model, ClassificationModel
from .datasets import get_classification_dataset, create_dataloader


class ImageClassificationTask:
    """
    Unified task interface for image classification.

    Handles training, evaluation, and inference for classification models.
    """

    def __init__(
        self,
        model: Optional[ClassificationModel] = None,
        architecture: str = "resnet",
        variant: str = "50",
        num_classes: int = 10,
        pretrained: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **model_kwargs
    ):
        """
        Args:
            model: Pre-initialized model (if None, will be created)
            architecture: Model architecture
            variant: Model variant
            num_classes: Number of classes
            pretrained: Whether to use pretrained weights
            device: Device to run on
        """
        self.device = device
        self.num_classes = num_classes

        # Create or use provided model
        if model is None:
            self.model = get_classification_model(
                architecture=architecture,
                variant=variant,
                num_classes=num_classes,
                pretrained=pretrained,
                **model_kwargs
            )
        else:
            self.model = model

        self.model = self.model.to(device)

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

    def prepare_training(
        self,
        optimizer: Literal["adam", "adamw", "sgd"] = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.9,
        scheduler: Optional[Literal["cosine", "step", "plateau"]] = "cosine",
        criterion: Optional[nn.Module] = None,
        **optimizer_kwargs
    ):
        """
        Prepare optimizer, scheduler, and loss function.

        Args:
            optimizer: Optimizer type
            lr: Learning rate
            weight_decay: Weight decay
            momentum: Momentum (for SGD)
            scheduler: LR scheduler type
            criterion: Loss function (default: CrossEntropyLoss)
        """
        # Setup optimizer
        if optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                **optimizer_kwargs
            )

        # Setup scheduler
        if scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100
            )
        elif scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=10
            )

        # Setup criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            use_amp: Whether to use automatic mixed precision

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': acc
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation data loader
            return_predictions: Whether to return predictions

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []

        pbar = tqdm(val_loader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': total_loss / len(val_loader),
                'acc': acc
            })

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        result = {'loss': avg_loss, 'accuracy': accuracy}

        if return_predictions:
            result['predictions'] = np.array(all_predictions)
            result['targets'] = np.array(all_targets)

        return result

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        use_amp: bool = False,
        save_dir: Optional[str] = None,
        early_stopping_patience: int = 0,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            use_amp: Whether to use AMP
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping (0 = disabled)
        """
        if self.optimizer is None:
            self.prepare_training()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, use_amp)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])

            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])

                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")

                # Save best model
                if val_metrics['accuracy'] > self.best_acc:
                    self.best_acc = val_metrics['accuracy']
                    if save_dir:
                        self.save_checkpoint(save_dir / "best_model.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['accuracy'])
                    else:
                        self.scheduler.step()

        # Save final model
        if save_dir:
            self.save_checkpoint(save_dir / "final_model.pth")
            self.save_history(save_dir / "history.json")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'best_acc': self.best_acc,
            'history': self.history,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.history = checkpoint.get('history', {})
        print(f"Checkpoint loaded from {path}")

    def save_history(self, path: str):
        """Save training history."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, return_probs: bool = False) -> np.ndarray:
        """
        Make predictions on images.

        Args:
            images: Input images [B, C, H, W]
            return_probs: Whether to return probabilities

        Returns:
            Predictions or probabilities
        """
        self.model.eval()
        images = images.to(self.device)

        outputs = self.model(images)

        if return_probs:
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
        else:
            _, predicted = outputs.max(1)
            return predicted.cpu().numpy()
