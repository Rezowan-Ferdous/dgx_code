"""
Logging utilities for the training framework
Supports console logging, file logging, and TensorBoard
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Unified logger supporting console, file, and TensorBoard logging
    """

    def __init__(
        self,
        name: str = "experiment",
        log_dir: Optional[str] = None,
        use_tensorboard: bool = True,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        """
        Initialize logger

        Args:
            name: Logger name / experiment name
            log_dir: Directory for log files
            use_tensorboard: Whether to use TensorBoard
            console_level: Logging level for console
            file_level: Logging level for file
        """
        self.name = name
        self.log_dir = log_dir

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir,
                f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.info(f"Logging to file: {log_file}")

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and log_dir is not None:
            tb_dir = os.path.join(log_dir, "tensorboard")
            self.writer = SummaryWriter(tb_dir)
            self.info(f"TensorBoard logging to: {tb_dir}")

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """
        Log metrics to console and TensorBoard

        Args:
            metrics: Dictionary of metric names and values
            step: Global step / epoch number
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Step {step} - {prefix}{metrics_str}")

        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}{key}", value, step)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to TensorBoard"""
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor, step: int):
        """Log image to TensorBoard"""
        if self.writer is not None:
            self.writer.add_image(tag, img_tensor, step)

    def log_figure(self, tag: str, figure, step: int):
        """Log matplotlib figure to TensorBoard"""
        if self.writer is not None:
            self.writer.add_figure(tag, figure, step)

    def log_hyperparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """Log hyperparameters to TensorBoard"""
        if self.writer is not None:
            self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        """Close the logger and TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
