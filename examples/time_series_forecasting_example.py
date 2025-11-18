"""
Example: Time Series Forecasting with Transformer

This script demonstrates how to:
1. Create a time series forecasting task
2. Generate synthetic data
3. Train a Transformer for forecasting
4. Evaluate and make predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from domains.time_series.forecasting import (
    TimeSeriesForecastingTask,
    get_forecasting_dataset
)


def main():
    # Configuration
    CONFIG = {
        'model_type': 'transformer',
        'input_dim': 1,
        'output_dim': 1,
        'forecast_length': 10,
        'backcast_length': 50,
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 1e-3,
    }

    print("=" * 50)
    print("Time Series Forecasting Example")
    print("=" * 50)

    # Create task
    print("\n1. Creating Transformer forecasting model...")
    task = TimeSeriesForecastingTask(
        model_type=CONFIG['model_type'],
        input_dim=CONFIG['input_dim'],
        output_dim=CONFIG['output_dim'],
        forecast_length=CONFIG['forecast_length'],
        backcast_length=CONFIG['backcast_length'],
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    print(f"✓ Model created: {CONFIG['model_type']}")

    # Prepare datasets
    print("\n2. Loading synthetic time series data...")
    train_dataset = get_forecasting_dataset(
        'synthetic',
        split='train',
        seq_length=CONFIG['backcast_length'],
        forecast_length=CONFIG['forecast_length']
    )
    val_dataset = get_forecasting_dataset(
        'synthetic',
        split='val',
        seq_length=CONFIG['backcast_length'],
        forecast_length=CONFIG['forecast_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False
    )
    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train
    print("\n3. Starting training...")
    task.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate']
    )
    print("✓ Training completed")

    # Evaluate
    print("\n4. Final evaluation...")
    metrics = task.evaluate(val_loader)
    print(f"✓ MSE: {metrics['mse']:.4f}")
    print(f"✓ RMSE: {metrics['rmse']:.4f}")

    # Make predictions
    print("\n5. Making predictions...")
    sample_batch = next(iter(val_loader))
    src, tgt = sample_batch

    predictions = task.predict(src[:1])  # Predict for first sample

    # Visualize
    print("\n6. Visualizing predictions...")
    plt.figure(figsize=(12, 6))

    # Historical data
    historical = src[0].numpy().flatten()
    plt.plot(range(len(historical)), historical, 'b-', label='Historical', linewidth=2)

    # True future
    true_future = tgt[0].numpy().flatten()
    plt.plot(range(len(historical), len(historical) + len(true_future)),
             true_future, 'g-', label='True Future', linewidth=2)

    # Predicted future
    pred_future = predictions[0].flatten()
    plt.plot(range(len(historical), len(historical) + len(pred_future)),
             pred_future, 'r--', label='Predicted Future', linewidth=2)

    plt.axvline(x=len(historical), color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Time Series Forecasting: Transformer Model')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = './forecasting_prediction.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
