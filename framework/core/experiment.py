"""
Experiment manager that ties together all components
Provides a high-level interface for running experiments
"""

import os
import torch
import random
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from ..config.base_config import ExperimentConfig
from ..utils.logger import Logger
from ..utils.checkpoint import CheckpointManager
from ..utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from .trainer import Trainer
from .tester import Tester
from .evaluator import Evaluator


class ExperimentManager:
    """
    High-level experiment manager
    Handles experiment setup, training, testing, evaluation, and reporting
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment manager

        Args:
            config: Experiment configuration
        """
        self.config = config

        # Set random seeds for reproducibility
        self._set_random_seeds(config.seed)

        # Create directories
        config.create_directories()

        # Initialize logger
        self.logger = Logger(
            name=config.experiment_name,
            log_dir=config.log_dir,
            use_tensorboard=config.visualization.use_tensorboard,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            save_best_only=config.training.save_best_only,
            logger=self.logger,
        )

        # Save configuration
        config_path = os.path.join(config.output_dir, config.experiment_name, "config.yaml")
        config.save(config_path)
        self.logger.info(f"Configuration saved to: {config_path}")

        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.trainer = None
        self.tester = None
        self.evaluator = None

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setup_model(self, model: Optional[torch.nn.Module] = None):
        """
        Setup model

        Args:
            model: Model instance (if None, will try to build from registry)
        """
        if model is None:
            # Try to build from registry
            if self.config.model.name in MODEL_REGISTRY:
                model_class = MODEL_REGISTRY.get(self.config.model.name)
                model = model_class(
                    num_classes=self.config.model.num_classes,
                    in_channel=self.config.model.in_channel,
                    n_features=self.config.model.n_features,
                    n_layers=self.config.model.n_layers,
                    **self.config.model.extra_params
                )
                self.logger.info(f"Built model from registry: {self.config.model.name}")
            else:
                raise ValueError(
                    f"Model {self.config.model.name} not found in registry. "
                    f"Available models: {MODEL_REGISTRY.list()}"
                )

        self.model = model.to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {num_params:,}")

    def setup_data(
        self,
        train_loader=None,
        val_loader=None,
        test_loader=None
    ):
        """
        Setup data loaders

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if train_loader:
            self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        if test_loader:
            self.logger.info(f"Test samples: {len(test_loader.dataset)}")

    def setup_optimizer(self, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Setup optimizer

        Args:
            optimizer: Optimizer instance (if None, will create from config)
        """
        if optimizer is None:
            if self.model is None:
                raise ValueError("Model must be set up before optimizer")

            if self.config.optimizer.name == "Adam":
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                )
            elif self.config.optimizer.name == "SGD":
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.config.optimizer.learning_rate,
                    momentum=self.config.optimizer.momentum,
                    weight_decay=self.config.optimizer.weight_decay,
                )
            elif self.config.optimizer.name == "AdamW":
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer.name}")

            self.logger.info(f"Created optimizer: {self.config.optimizer.name}")

        self.optimizer = optimizer

    def setup_criterion(self, criterion: Optional[torch.nn.Module] = None):
        """
        Setup loss criterion

        Args:
            criterion: Loss function instance
        """
        if criterion is None:
            # Default to CrossEntropyLoss
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
            self.logger.info("Using default CrossEntropyLoss")

        self.criterion = criterion

    def setup_trainer(self):
        """Setup trainer"""
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise ValueError("Model, optimizer, and criterion must be set up before trainer")

        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Train and validation loaders must be set up before trainer")

        self.trainer = Trainer(
            config=self.config,
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            checkpoint_manager=self.checkpoint_manager,
        )

        self.logger.info("Trainer initialized")

    def setup_tester(self):
        """Setup tester"""
        if self.model is None:
            raise ValueError("Model must be set up before tester")

        if self.test_loader is None:
            raise ValueError("Test loader must be set up before tester")

        self.tester = Tester(
            config=self.config,
            model=self.model,
            test_loader=self.test_loader,
            device=self.device,
            logger=self.logger,
        )

        self.logger.info("Tester initialized")

    def setup_evaluator(self):
        """Setup evaluator"""
        self.evaluator = Evaluator(
            config=self.config,
            logger=self.logger,
        )

        self.logger.info("Evaluator initialized")

    def train(self) -> Dict[str, Any]:
        """
        Run training

        Returns:
            Training history
        """
        if self.trainer is None:
            self.setup_trainer()

        self.logger.info("=" * 80)
        self.logger.info("Starting Training")
        self.logger.info("=" * 80)

        history = self.trainer.train()

        self.logger.info("=" * 80)
        self.logger.info("Training Completed")
        self.logger.info("=" * 80)

        return history

    def test(self, load_best: bool = True) -> Dict[str, Any]:
        """
        Run testing

        Args:
            load_best: Whether to load best model before testing

        Returns:
            Predictions dictionary
        """
        if self.tester is None:
            self.setup_tester()

        # Load best model if requested
        if load_best:
            best_checkpoint = self.checkpoint_manager.get_best_checkpoint_path()
            if best_checkpoint:
                self.logger.info(f"Loading best model from: {best_checkpoint}")
                self.checkpoint_manager.load_checkpoint(
                    best_checkpoint,
                    self.model,
                    device=self.device
                )

        self.logger.info("=" * 80)
        self.logger.info("Starting Testing")
        self.logger.info("=" * 80)

        predictions = self.tester.predict()

        # Save predictions
        if self.config.reporting.save_predictions:
            pred_path = os.path.join(
                self.config.output_dir,
                self.config.experiment_name,
                "predictions",
                "predictions.npz"
            )
            self.tester.save_predictions(predictions, pred_path)

        self.logger.info("=" * 80)
        self.logger.info("Testing Completed")
        self.logger.info("=" * 80)

        return predictions

    def evaluate(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Run evaluation

        Args:
            predictions: Predictions dictionary from tester

        Returns:
            Metrics dictionary
        """
        if self.evaluator is None:
            self.setup_evaluator()

        self.logger.info("=" * 80)
        self.logger.info("Starting Evaluation")
        self.logger.info("=" * 80)

        metrics = self.evaluator.evaluate(
            predictions=predictions['predictions'],
            ground_truth=predictions['labels'],
        )

        # Save metrics
        if self.config.reporting.save_metrics_csv:
            import pandas as pd
            metrics_df = pd.DataFrame([metrics])
            metrics_path = os.path.join(
                self.config.output_dir,
                self.config.experiment_name,
                "metrics.csv"
            )
            metrics_df.to_csv(metrics_path, index=False)
            self.logger.info(f"Metrics saved to: {metrics_path}")

        self.logger.info("=" * 80)
        self.logger.info("Evaluation Completed")
        self.logger.info("=" * 80)

        return metrics

    def run_full_experiment(self):
        """
        Run complete experiment: train -> test -> evaluate -> report

        Returns:
            Dictionary with all results
        """
        results = {}

        # Training
        if self.train_loader is not None:
            history = self.train()
            results['history'] = history

        # Testing
        if self.test_loader is not None:
            predictions = self.test(load_best=True)
            results['predictions'] = predictions

            # Evaluation
            metrics = self.evaluate(predictions)
            results['metrics'] = metrics

        return results

    def close(self):
        """Close logger and clean up"""
        self.logger.close()
