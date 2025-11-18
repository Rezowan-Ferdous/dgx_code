"""Label encoding utilities"""

import torch
import torch.nn as nn
from typing import List, Optional, Union
import numpy as np


class LabelEncoder:
    """
    Encode and decode labels for multi-class and multi-label tasks
    """

    def __init__(
        self,
        classes: List[str],
        multi_label: bool = False
    ):
        """
        Initialize label encoder

        Args:
            classes: List of class names
            multi_label: Whether this is multi-label classification
        """
        self.classes = classes
        self.num_classes = len(classes)
        self.multi_label = multi_label

        # Create mappings
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

    def encode(
        self,
        labels: Union[str, List[str], int, List[int]]
    ) -> Union[int, List[int], np.ndarray]:
        """
        Encode labels to indices

        Args:
            labels: Class name(s) or index/indices

        Returns:
            Encoded label indices
        """
        if isinstance(labels, str):
            return self.class_to_idx[labels]
        elif isinstance(labels, list):
            if isinstance(labels[0], str):
                return [self.class_to_idx[label] for label in labels]
            else:
                return labels  # Already indices
        else:
            return labels  # Single index

    def decode(
        self,
        indices: Union[int, List[int], np.ndarray, torch.Tensor]
    ) -> Union[str, List[str]]:
        """
        Decode indices to class names

        Args:
            indices: Class indices

        Returns:
            Class names
        """
        if isinstance(indices, (np.ndarray, torch.Tensor)):
            indices = indices.tolist() if hasattr(indices, 'tolist') else list(indices)

        if isinstance(indices, int):
            return self.idx_to_class[indices]
        elif isinstance(indices, list):
            return [self.idx_to_class[idx] for idx in indices]
        else:
            return self.idx_to_class[int(indices)]

    def one_hot_encode(
        self,
        indices: Union[int, List[int], np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Convert indices to one-hot encoding

        Args:
            indices: Class indices

        Returns:
            One-hot encoded array
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        elif isinstance(indices, list):
            indices = np.array(indices)
        elif isinstance(indices, int):
            indices = np.array([indices])

        one_hot = np.zeros((len(indices), self.num_classes))
        one_hot[np.arange(len(indices)), indices] = 1

        return one_hot

    def one_hot_decode(
        self,
        one_hot: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Convert one-hot encoding to indices

        Args:
            one_hot: One-hot encoded array [B, num_classes]

        Returns:
            Class indices [B]
        """
        if isinstance(one_hot, torch.Tensor):
            one_hot = one_hot.cpu().numpy()

        return np.argmax(one_hot, axis=1)

    def get_class_weights(
        self,
        label_counts: Optional[np.ndarray] = None,
        method: str = "inverse"  # 'inverse' or 'balanced'
    ) -> np.ndarray:
        """
        Compute class weights for imbalanced data

        Args:
            label_counts: Count of samples per class
            method: Weighting method

        Returns:
            Class weights
        """
        if label_counts is None:
            return np.ones(self.num_classes)

        if method == "inverse":
            weights = 1.0 / (label_counts + 1e-6)
        elif method == "balanced":
            total = np.sum(label_counts)
            weights = total / (self.num_classes * label_counts + 1e-6)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Normalize
        weights = weights / weights.sum() * self.num_classes

        return weights
