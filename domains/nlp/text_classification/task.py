"""
Text Classification Task
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm

from .models import get_text_classification_model, TextClassificationModel
from .datasets import get_text_classification_dataset


class TextClassificationTask:
    """Unified interface for text classification."""

    def __init__(
        self,
        model: Optional[TextClassificationModel] = None,
        architecture: str = "bert",
        variant: str = "base",
        num_classes: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **model_kwargs
    ):
        self.device = device
        self.num_classes = num_classes

        if model is None:
            self.model = get_text_classification_model(
                architecture=architecture,
                variant=variant,
                num_classes=num_classes,
                **model_kwargs
            )
        else:
            self.model = model

        self.model = self.model.to(device)
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 3,
        lr: float = 2e-5,
    ):
        """Full training loop."""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%")

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.2f}%")

    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on texts."""
        self.model.eval()

        encodings = self.model.encode_texts(texts)
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predictions = logits.argmax(dim=-1)

        return predictions.cpu().numpy()
