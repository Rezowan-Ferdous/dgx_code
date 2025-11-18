"""
Image Classification Module

State-of-the-art models and datasets for image classification tasks.
"""

from .models import get_classification_model, ClassificationModel
from .datasets import get_classification_dataset, ClassificationDataset
from .task import ImageClassificationTask

__all__ = [
    'get_classification_model',
    'ClassificationModel',
    'get_classification_dataset',
    'ClassificationDataset',
    'ImageClassificationTask',
]
