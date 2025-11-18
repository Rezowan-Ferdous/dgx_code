"""
Unified Task Runner

Central interface for running any task (vision, time series, or NLP) with state-of-the-art models.

This module provides a unified API to:
- Run classification, detection, segmentation, and other vision tasks
- Perform time series forecasting, classification, and anomaly detection
- Execute text classification, NER, QA, and generation tasks
"""

import torch
from typing import Dict, Any, Optional, Literal
from pathlib import Path


class TaskRunner:
    """
    Unified task runner for all domains.

    This class provides a simple interface to run any supported task
    with state-of-the-art models and datasets.
    """

    def __init__(
        self,
        domain: Literal["vision", "time_series", "nlp"],
        task: str,
        model_architecture: Optional[str] = None,
        model_variant: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize task runner.

        Args:
            domain: Task domain ('vision', 'time_series', 'nlp')
            task: Specific task within domain
            model_architecture: Model architecture to use
            model_variant: Model variant
            device: Device to run on
            **kwargs: Additional task-specific arguments

        Examples:
            >>> # Vision: Image classification
            >>> runner = TaskRunner(
            ...     domain='vision',
            ...     task='classification',
            ...     model_architecture='resnet',
            ...     model_variant='50'
            ... )
            >>>
            >>> # Time Series: Forecasting
            >>> runner = TaskRunner(
            ...     domain='time_series',
            ...     task='forecasting',
            ...     model_architecture='lstm'
            ... )
            >>>
            >>> # NLP: Text classification
            >>> runner = TaskRunner(
            ...     domain='nlp',
            ...     task='text_classification',
            ...     model_architecture='bert',
            ...     model_variant='base'
            ... )
        """
        self.domain = domain
        self.task = task
        self.device = device

        # Initialize task based on domain
        if domain == "vision":
            self._init_vision_task(task, model_architecture, model_variant, **kwargs)
        elif domain == "time_series":
            self._init_time_series_task(task, model_architecture, model_variant, **kwargs)
        elif domain == "nlp":
            self._init_nlp_task(task, model_architecture, model_variant, **kwargs)
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _init_vision_task(self, task: str, architecture: str, variant: str, **kwargs):
        """Initialize vision task."""
        if task == "classification":
            from domains.computer_vision.classification import ImageClassificationTask

            architecture = architecture or "resnet"
            variant = variant or "50"

            self.task_instance = ImageClassificationTask(
                architecture=architecture,
                variant=variant,
                device=self.device,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown vision task: {task}")

    def _init_time_series_task(self, task: str, architecture: str, variant: str, **kwargs):
        """Initialize time series task."""
        if task == "forecasting":
            from domains.time_series.forecasting import TimeSeriesForecastingTask

            model_type = architecture or "lstm"

            self.task_instance = TimeSeriesForecastingTask(
                model_type=model_type,
                device=self.device,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown time series task: {task}")

    def _init_nlp_task(self, task: str, architecture: str, variant: str, **kwargs):
        """Initialize NLP task."""
        if task == "text_classification":
            from domains.nlp.text_classification import TextClassificationTask

            architecture = architecture or "bert"
            variant = variant or "base"

            self.task_instance = TextClassificationTask(
                architecture=architecture,
                variant=variant,
                device=self.device,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown NLP task: {task}")

    def prepare_data(
        self,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        **dataset_kwargs
    ) -> Dict[str, Any]:
        """
        Prepare datasets and dataloaders.

        Args:
            dataset_name: Name of dataset to use
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            **dataset_kwargs: Additional dataset arguments

        Returns:
            Dictionary with train/val/test loaders
        """
        from torch.utils.data import DataLoader

        loaders = {}

        if self.domain == "vision":
            from domains.computer_vision.classification import (
                get_classification_dataset,
                create_dataloader
            )

            # Create datasets
            train_dataset = get_classification_dataset(
                dataset_name=dataset_name,
                split='train',
                **dataset_kwargs
            )
            val_dataset = get_classification_dataset(
                dataset_name=dataset_name,
                split='val',
                **dataset_kwargs
            )

            # Create loaders
            loaders['train'] = create_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            loaders['val'] = create_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

        elif self.domain == "time_series":
            from domains.time_series.forecasting import get_forecasting_dataset

            train_dataset = get_forecasting_dataset(
                dataset_name=dataset_name,
                split='train',
                **dataset_kwargs
            )
            val_dataset = get_forecasting_dataset(
                dataset_name=dataset_name,
                split='val',
                **dataset_kwargs
            )

            loaders['train'] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

        elif self.domain == "nlp":
            from domains.nlp.text_classification import get_text_classification_dataset

            tokenizer = self.task_instance.model.tokenizer

            train_dataset = get_text_classification_dataset(
                dataset_name=dataset_name,
                split='train',
                tokenizer=tokenizer,
                **dataset_kwargs
            )

            loaders['train'] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

        return loaders

    def train(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        **train_kwargs
    ):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            **train_kwargs: Additional training arguments
        """
        self.task_instance.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            **train_kwargs
        )

    def evaluate(self, eval_loader, **eval_kwargs):
        """Evaluate the model."""
        return self.task_instance.evaluate(eval_loader, **eval_kwargs)

    def predict(self, inputs, **predict_kwargs):
        """Make predictions."""
        return self.task_instance.predict(inputs, **predict_kwargs)

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.task_instance.save_checkpoint(path)

    def load(self, path: str):
        """Load model checkpoint."""
        self.task_instance.load_checkpoint(path)


# Convenience functions for quick task setup
def run_vision_task(
    task: str = "classification",
    model: str = "resnet50",
    dataset: str = "cifar10",
    epochs: int = 10,
    batch_size: int = 32,
    **kwargs
):
    """Quick setup for vision tasks."""
    architecture, variant = model.split("_") if "_" in model else (model, "50")

    runner = TaskRunner(
        domain="vision",
        task=task,
        model_architecture=architecture.replace(variant, ""),
        model_variant=variant,
        **kwargs
    )

    loaders = runner.prepare_data(dataset, batch_size=batch_size)
    runner.train(loaders['train'], loaders.get('val'), epochs=epochs)

    return runner


def run_time_series_task(
    task: str = "forecasting",
    model: str = "lstm",
    dataset: str = "synthetic",
    epochs: int = 10,
    batch_size: int = 32,
    **kwargs
):
    """Quick setup for time series tasks."""
    runner = TaskRunner(
        domain="time_series",
        task=task,
        model_architecture=model,
        **kwargs
    )

    loaders = runner.prepare_data(dataset, batch_size=batch_size)
    runner.train(loaders['train'], loaders.get('val'), epochs=epochs)

    return runner


def run_nlp_task(
    task: str = "text_classification",
    model: str = "bert_base",
    dataset: str = "imdb",
    epochs: int = 3,
    batch_size: int = 16,
    **kwargs
):
    """Quick setup for NLP tasks."""
    architecture, variant = model.split("_") if "_" in model else (model, "base")

    runner = TaskRunner(
        domain="nlp",
        task=task,
        model_architecture=architecture,
        model_variant=variant,
        **kwargs
    )

    loaders = runner.prepare_data(dataset, batch_size=batch_size)
    runner.train(loaders['train'], loaders.get('val'), epochs=epochs)

    return runner


# Example configurations
EXAMPLE_CONFIGS = {
    "vision_classification_cifar10": {
        "domain": "vision",
        "task": "classification",
        "model": "resnet50",
        "dataset": "cifar10",
        "epochs": 100,
        "batch_size": 128,
    },
    "vision_classification_imagenet": {
        "domain": "vision",
        "task": "classification",
        "model": "efficientnet_b0",
        "dataset": "imagenet",
        "epochs": 300,
        "batch_size": 256,
    },
    "time_series_forecasting": {
        "domain": "time_series",
        "task": "forecasting",
        "model": "transformer",
        "dataset": "electricity",
        "epochs": 50,
        "batch_size": 32,
    },
    "nlp_sentiment_analysis": {
        "domain": "nlp",
        "task": "text_classification",
        "model": "roberta_base",
        "dataset": "imdb",
        "epochs": 3,
        "batch_size": 16,
    },
}


def run_example(config_name: str):
    """Run a predefined example configuration."""
    if config_name not in EXAMPLE_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")

    config = EXAMPLE_CONFIGS[config_name]
    domain = config.pop('domain')
    model = config.pop('model')
    dataset = config.pop('dataset')
    task = config.pop('task')
    epochs = config.pop('epochs')
    batch_size = config.pop('batch_size')

    if domain == "vision":
        return run_vision_task(task, model, dataset, epochs, batch_size, **config)
    elif domain == "time_series":
        return run_time_series_task(task, model, dataset, epochs, batch_size, **config)
    elif domain == "nlp":
        return run_nlp_task(task, model, dataset, epochs, batch_size, **config)
