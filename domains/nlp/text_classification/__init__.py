"""
Text Classification Module

State-of-the-art models for text classification tasks.
"""

from .models import get_text_classification_model, TextClassificationModel
from .datasets import get_text_classification_dataset
from .task import TextClassificationTask

__all__ = [
    'get_text_classification_model',
    'TextClassificationModel',
    'get_text_classification_dataset',
    'TextClassificationTask',
]
