"""
Time Series Datasets for Forecasting

Popular datasets and data loaders for time series forecasting.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """Time series dataset for forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        seq_length: int = 20,
        forecast_length: int = 10,
        stride: int = 1,
        normalize: bool = True,
    ):
        """
        Args:
            data: Time series data [time_steps, features]
            seq_length: Input sequence length
            forecast_length: Forecast horizon
            stride: Stride for creating sequences
            normalize: Whether to normalize data
        """
        self.data = data
        self.seq_length = seq_length
        self.forecast_length = forecast_length
        self.stride = stride

        # Normalize
        if normalize:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8
            self.data = (data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        # Create sequences
        self.sequences = []
        for i in range(0, len(data) - seq_length - forecast_length + 1, stride):
            src = data[i:i + seq_length]
            tgt = data[i + seq_length:i + seq_length + forecast_length]
            self.sequences.append((src, tgt))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.sequences[idx]
        return torch.FloatTensor(src), torch.FloatTensor(tgt)

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        if self.mean is not None:
            return data * self.std + self.mean
        return data


def get_forecasting_dataset(
    dataset_name: str = "synthetic",
    root: str = "./data",
    split: str = "train",
    seq_length: int = 20,
    forecast_length: int = 10,
    **kwargs
) -> TimeSeriesDataset:
    """
    Factory function for time series datasets.

    Args:
        dataset_name: Name of dataset ('synthetic', 'electricity', 'traffic', 'weather')
        root: Data directory
        split: Dataset split
        seq_length: Input sequence length
        forecast_length: Forecast horizon

    Returns:
        TimeSeriesDataset instance
    """

    if dataset_name == "synthetic":
        # Generate synthetic sine wave data
        t = np.linspace(0, 100, 10000)
        data = np.sin(t) + 0.5 * np.sin(5 * t) + 0.1 * np.random.randn(len(t))
        data = data.reshape(-1, 1)

        # Split
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))

        if split == "train":
            data = data[:train_size]
        elif split == "val":
            data = data[train_size:train_size + val_size]
        else:
            data = data[train_size + val_size:]

    elif dataset_name in ["electricity", "traffic", "weather"]:
        # Placeholder for real datasets
        # In practice, load from CSV files
        data_path = Path(root) / dataset_name / f"{split}.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            data = df.values
        else:
            # Generate placeholder data
            data = np.random.randn(1000, 1)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return TimeSeriesDataset(
        data=data,
        seq_length=seq_length,
        forecast_length=forecast_length,
        **kwargs
    )


# Dataset information
DATASET_INFO = {
    "electricity": {
        "description": "Electricity consumption data",
        "num_features": 321,
        "frequency": "hourly",
    },
    "traffic": {
        "description": "Traffic data from California highways",
        "num_features": 862,
        "frequency": "hourly",
    },
    "weather": {
        "description": "Weather data with multiple variables",
        "num_features": 21,
        "frequency": "10min",
    },
}
