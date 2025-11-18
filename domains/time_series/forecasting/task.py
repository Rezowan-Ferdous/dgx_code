"""
Time Series Forecasting Task
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np
from tqdm import tqdm

from .models import get_forecasting_model, ForecastingModel
from .datasets import get_forecasting_dataset, TimeSeriesDataset


class TimeSeriesForecastingTask:
    """Unified interface for time series forecasting."""

    def __init__(
        self,
        model: Optional[ForecastingModel] = None,
        model_type: str = "lstm",
        input_dim: int = 1,
        output_dim: int = 1,
        forecast_length: int = 10,
        backcast_length: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **model_kwargs
    ):
        self.device = device
        self.forecast_length = forecast_length

        if model is None:
            self.model = get_forecasting_model(
                model_type=model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                forecast_length=forecast_length,
                backcast_length=backcast_length,
                **model_kwargs
            )
        else:
            self.model = model

        self.model = self.model.to(device)
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for src, tgt in tqdm(train_loader, desc="Training"):
            src, tgt = src.to(self.device), tgt.to(self.device)

            self.optimizer.zero_grad()

            if self.model.model_type == "transformer":
                # Transformer needs decoder input
                dec_input = torch.zeros_like(tgt)
                output = self.model(src, dec_input)
            else:
                output = self.model(src)

            # Ensure output shape matches target
            if output.shape != tgt.shape:
                if len(output.shape) == 2:
                    output = output.unsqueeze(-1)
                output = output[:, -self.forecast_length:]

            loss = self.criterion(output, tgt)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for src, tgt in val_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)

            if self.model.model_type == "transformer":
                dec_input = torch.zeros_like(tgt)
                output = self.model(src, dec_input)
            else:
                output = self.model(src)

            if output.shape != tgt.shape:
                if len(output.shape) == 2:
                    output = output.unsqueeze(-1)
                output = output[:, -self.forecast_length:]

            loss = self.criterion(output, tgt)
            total_loss += loss.item()

            all_preds.append(output.cpu())
            all_targets.append(tgt.cpu())

        mse = total_loss / len(val_loader)
        rmse = np.sqrt(mse)

        return {'mse': mse, 'rmse': rmse}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 1e-3,
    ):
        """Full training loop."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

            if val_loader:
                metrics = self.evaluate(val_loader)
                print(f"Val MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        x = x.to(self.device)

        if self.model.model_type == "transformer":
            tgt = torch.zeros(x.size(0), self.forecast_length, x.size(-1)).to(self.device)
            output = self.model(x, tgt)
        else:
            output = self.model(x)

        return output.cpu().numpy()
