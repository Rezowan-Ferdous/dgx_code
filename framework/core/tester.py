"""
Testing module for the framework
Handles model inference and prediction generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Any
from tqdm import tqdm

from ..config.base_config import ExperimentConfig
from ..utils.logger import Logger


class Tester:
    """
    Testing class for model inference and prediction generation
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        test_loader,
        device: str = "cuda",
        logger: Optional[Logger] = None,
    ):
        """
        Initialize tester

        Args:
            config: Experiment configuration
            model: Model to test
            test_loader: Test data loader
            device: Device to run inference on
            logger: Logger instance
        """
        self.config = config
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.logger = logger

    def predict(self, return_features: bool = False) -> Dict[str, Any]:
        """
        Generate predictions on test set

        Args:
            return_features: Whether to return model features

        Returns:
            Dictionary containing predictions, labels, and optionally features
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_video_ids = []
        all_probabilities = []
        all_features = [] if return_features else None

        if self.logger:
            self.logger.info("Generating predictions...")

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", disable=self.logger is None):
                # Get data
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch.get('mask', None)
                video_id = batch.get('video_id', None)

                if mask is not None:
                    mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(features, mask)

                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                # Move to CPU and convert to numpy
                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()
                probabilities = probabilities.cpu().numpy()

                if mask is not None:
                    mask = mask.cpu().numpy()
                    # Apply mask to predictions
                    for i in range(len(predictions)):
                        valid_idx = mask[i] > 0
                        all_predictions.append(predictions[i][valid_idx])
                        all_labels.append(labels[i][valid_idx])
                        all_probabilities.append(probabilities[i][:, valid_idx])
                else:
                    for i in range(len(predictions)):
                        all_predictions.append(predictions[i])
                        all_labels.append(labels[i])
                        all_probabilities.append(probabilities[i])

                # Store video IDs
                if video_id is not None:
                    if isinstance(video_id, torch.Tensor):
                        video_id = video_id.cpu().numpy()
                    all_video_ids.extend(video_id)

                # Store features if requested
                if return_features:
                    all_features.append(features.cpu().numpy())

        results = {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'video_ids': all_video_ids if all_video_ids else None,
        }

        if return_features:
            results['features'] = all_features

        if self.logger:
            self.logger.info(f"Generated predictions for {len(all_predictions)} videos")

        return results

    def predict_single(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
        """
        Generate prediction for a single sample

        Args:
            features: Input features [T, D]
            mask: Optional mask [T]

        Returns:
            Dictionary with predictions and probabilities
        """
        self.model.eval()

        # Add batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        features = features.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(features, mask)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

        # Remove batch dimension and convert to numpy
        predictions = predictions[0].cpu().numpy()
        probabilities = probabilities[0].cpu().numpy()

        return {
            'predictions': predictions,
            'probabilities': probabilities,
        }

    def save_predictions(
        self,
        predictions: Dict[str, Any],
        save_path: str,
        format: str = 'npz'
    ):
        """
        Save predictions to file

        Args:
            predictions: Dictionary containing predictions
            save_path: Path to save predictions
            format: Save format ('npz', 'pkl', or 'txt')
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if format == 'npz':
            # Save as NumPy archive
            np.savez(
                save_path,
                predictions=predictions['predictions'],
                labels=predictions['labels'],
                probabilities=predictions['probabilities'],
                video_ids=predictions.get('video_ids', None),
            )
        elif format == 'pkl':
            # Save as pickle
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(predictions, f)
        elif format == 'txt':
            # Save predictions as text (one per line)
            with open(save_path, 'w') as f:
                for i, pred in enumerate(predictions['predictions']):
                    f.write(f"Video {i}: {' '.join(map(str, pred))}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")

        if self.logger:
            self.logger.info(f"Saved predictions to: {save_path}")

    def load_predictions(self, load_path: str, format: str = 'npz') -> Dict[str, Any]:
        """
        Load predictions from file

        Args:
            load_path: Path to load predictions from
            format: Load format ('npz' or 'pkl')

        Returns:
            Dictionary containing predictions
        """
        if format == 'npz':
            data = np.load(load_path, allow_pickle=True)
            predictions = {
                'predictions': data['predictions'],
                'labels': data['labels'],
                'probabilities': data['probabilities'],
                'video_ids': data.get('video_ids', None),
            }
        elif format == 'pkl':
            import pickle
            with open(load_path, 'rb') as f:
                predictions = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if self.logger:
            self.logger.info(f"Loaded predictions from: {load_path}")

        return predictions
