"""
Computer Vision Tasks Module

This module provides state-of-the-art models and datasets for various computer vision tasks:
- Image Classification
- Object Detection
- Semantic/Instance Segmentation
- Pose Estimation
- Video Understanding
- Surgical Action Recognition
"""

from .classification import (
    ImageClassificationTask,
    get_classification_model,
    get_classification_dataset
)
from .object_detection import (
    ObjectDetectionTask,
    get_detection_model,
    get_detection_dataset
)
from .segmentation import (
    SegmentationTask,
    get_segmentation_model,
    get_segmentation_dataset
)
from .video_understanding import (
    VideoClassificationTask,
    ActionRecognitionTask,
    get_video_model,
    get_video_dataset
)

__all__ = [
    # Classification
    'ImageClassificationTask',
    'get_classification_model',
    'get_classification_dataset',

    # Detection
    'ObjectDetectionTask',
    'get_detection_model',
    'get_detection_dataset',

    # Segmentation
    'SegmentationTask',
    'get_segmentation_model',
    'get_segmentation_dataset',

    # Video
    'VideoClassificationTask',
    'ActionRecognitionTask',
    'get_video_model',
    'get_video_dataset',
]
