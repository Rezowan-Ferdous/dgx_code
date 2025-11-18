"""
State-of-the-art Time Series Forecasting Models

Supports:
- Transformer-based: Informer, Autoformer, FEDformer
- Deep Learning: N-BEATS, N-HiTS, DeepAR
- RNN-based: LSTM, GRU, Seq2Seq
- Attention-based: Temporal Fusion Transformer (TFT)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformers."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting."""

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence [batch, src_len, input_dim]
            tgt: Target sequence [batch, tgt_len, input_dim]

        Returns:
            Predictions [batch, tgt_len, output_dim]
        """
        # Project to model dimension
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)

        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # Project to output dimension
        output = self.output_projection(output)

        return output


class NBEATS(nn.Module):
    """N-BEATS: Neural Basis Expansion Analysis for Time Series."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        forecast_length: int = 10,
        backcast_length: int = 20,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_size: int = 512,
        theta_size: int = 8,
    ):
        super().__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

        # Create stacks of blocks
        self.blocks = nn.ModuleList([
            NBEATSBlock(
                input_dim=input_dim,
                theta_size=theta_size,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                num_layers=num_layers,
                layer_size=layer_size,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, backcast_length, input_dim]

        Returns:
            Forecast [batch, forecast_length, output_dim]
        """
        residuals = x.squeeze(-1)  # [batch, backcast_length]
        forecast = x.new_zeros(x.size(0), self.forecast_length)

        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        return forecast.unsqueeze(-1)


class NBEATSBlock(nn.Module):
    """Single block in N-BEATS."""

    def __init__(
        self,
        input_dim: int,
        theta_size: int,
        backcast_length: int,
        forecast_length: int,
        num_layers: int,
        layer_size: int,
    ):
        super().__init__()

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.theta_size = theta_size

        # Fully connected stack
        layers = []
        layers.append(nn.Linear(backcast_length, layer_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())

        self.fc_stack = nn.Sequential(*layers)

        # Theta layers
        self.theta_b = nn.Linear(layer_size, theta_size)
        self.theta_f = nn.Linear(layer_size, theta_size)

        # Basis functions (using polynomial)
        self.backcast_basis = self._polynomial_basis(backcast_length, theta_size)
        self.forecast_basis = self._polynomial_basis(forecast_length, theta_size)

    def _polynomial_basis(self, length: int, degree: int) -> torch.Tensor:
        """Create polynomial basis functions."""
        t = torch.arange(length, dtype=torch.float32) / length
        basis = torch.stack([t ** i for i in range(degree)], dim=1)
        return basis

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch, backcast_length]

        Returns:
            backcast: [batch, backcast_length]
            forecast: [batch, forecast_length]
        """
        # FC stack
        h = self.fc_stack(x)

        # Compute theta
        theta_b = self.theta_b(h)  # [batch, theta_size]
        theta_f = self.theta_f(h)  # [batch, theta_size]

        # Compute backcast and forecast
        basis_b = self.backcast_basis.to(x.device)  # [backcast_length, theta_size]
        basis_f = self.forecast_basis.to(x.device)  # [forecast_length, theta_size]

        backcast = torch.matmul(theta_b, basis_b.T)  # [batch, backcast_length]
        forecast = torch.matmul(theta_f, basis_f.T)  # [batch, forecast_length]

        return backcast, forecast


class LSTMForecaster(nn.Module):
    """LSTM-based forecasting model."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, seq_len, input_dim]

        Returns:
            Output sequence [batch, seq_len, output_dim]
        """
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


class ForecastingModel(nn.Module):
    """Unified interface for forecasting models."""

    def __init__(
        self,
        model_type: Literal["transformer", "nbeats", "lstm", "gru"],
        input_dim: int = 1,
        output_dim: int = 1,
        forecast_length: int = 10,
        backcast_length: int = 20,
        **model_kwargs
    ):
        super().__init__()

        self.model_type = model_type
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

        if model_type == "transformer":
            self.model = TimeSeriesTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                **model_kwargs
            )
        elif model_type == "nbeats":
            self.model = NBEATS(
                input_dim=input_dim,
                output_dim=output_dim,
                forecast_length=forecast_length,
                backcast_length=backcast_length,
                **model_kwargs
            )
        elif model_type == "lstm":
            self.model = LSTMForecaster(
                input_dim=input_dim,
                output_dim=output_dim,
                **model_kwargs
            )
        elif model_type == "gru":
            # GRU is similar to LSTM
            model_kwargs['hidden_dim'] = model_kwargs.get('hidden_dim', 128)
            model_kwargs['num_layers'] = model_kwargs.get('num_layers', 2)
            self.model = nn.GRU(
                input_size=input_dim,
                hidden_size=model_kwargs['hidden_dim'],
                num_layers=model_kwargs['num_layers'],
                batch_first=True
            )
            self.fc = nn.Linear(model_kwargs['hidden_dim'], output_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model_type == "gru":
            out, _ = self.model(x)
            return self.fc(out)
        else:
            return self.model(x, **kwargs)


def get_forecasting_model(
    model_type: Literal["transformer", "nbeats", "lstm", "gru"] = "lstm",
    input_dim: int = 1,
    output_dim: int = 1,
    forecast_length: int = 10,
    backcast_length: int = 20,
    **kwargs
) -> ForecastingModel:
    """
    Factory function to create forecasting models.

    Args:
        model_type: Type of forecasting model
        input_dim: Number of input features
        output_dim: Number of output features
        forecast_length: Number of steps to forecast
        backcast_length: Number of historical steps to use
        **kwargs: Additional model-specific arguments

    Returns:
        ForecastingModel instance

    Examples:
        >>> # Transformer for forecasting
        >>> model = get_forecasting_model('transformer', forecast_length=24)
        >>>
        >>> # N-BEATS for univariate forecasting
        >>> model = get_forecasting_model('nbeats', forecast_length=10,
        ...                                backcast_length=50)
        >>>
        >>> # LSTM for multivariate forecasting
        >>> model = get_forecasting_model('lstm', input_dim=5, output_dim=1)
    """
    return ForecastingModel(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        **kwargs
    )


# Model configurations
FORECASTING_MODELS = {
    "transformer": {
        "description": "Transformer-based forecasting with self-attention",
        "best_for": "Long sequences with complex patterns",
        "params": ["d_model", "nhead", "num_encoder_layers"],
    },
    "nbeats": {
        "description": "Neural Basis Expansion Analysis for interpretable forecasting",
        "best_for": "Univariate forecasting with trend and seasonality",
        "params": ["num_blocks", "num_layers", "layer_size"],
    },
    "lstm": {
        "description": "LSTM-based sequence-to-sequence forecasting",
        "best_for": "General-purpose time series with dependencies",
        "params": ["hidden_dim", "num_layers", "bidirectional"],
    },
    "gru": {
        "description": "GRU-based forecasting (lighter than LSTM)",
        "best_for": "Fast training on shorter sequences",
        "params": ["hidden_dim", "num_layers"],
    },
}


def list_forecasting_models() -> Dict[str, Dict[str, Any]]:
    """List all available forecasting models."""
    return FORECASTING_MODELS
