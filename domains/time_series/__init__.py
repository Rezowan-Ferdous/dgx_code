"""
Time Series Analysis Module

This module provides state-of-the-art models and datasets for time series tasks:
- Time Series Forecasting
- Time Series Classification
- Anomaly Detection
- Trend Analysis
"""

from .forecasting import (
    TimeSeriesForecastingTask,
    get_forecasting_model,
    get_forecasting_dataset
)
from .classification import (
    TimeSeriesClassificationTask,
    get_ts_classification_model,
    get_ts_classification_dataset
)
from .anomaly_detection import (
    AnomalyDetectionTask,
    get_anomaly_model,
    get_anomaly_dataset
)

__all__ = [
    # Forecasting
    'TimeSeriesForecastingTask',
    'get_forecasting_model',
    'get_forecasting_dataset',

    # Classification
    'TimeSeriesClassificationTask',
    'get_ts_classification_model',
    'get_ts_classification_dataset',

    # Anomaly Detection
    'AnomalyDetectionTask',
    'get_anomaly_model',
    'get_anomaly_dataset',
]
