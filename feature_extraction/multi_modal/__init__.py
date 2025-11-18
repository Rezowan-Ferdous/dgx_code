"""Multi-modal feature extractors for video, image, and text"""

from .video_extractor import VideoFeatureExtractor
from .image_extractor import ImageFeatureExtractor
from .text_extractor import TextFeatureExtractor
from .fusion import MultiModalFusion

__all__ = [
    'VideoFeatureExtractor',
    'ImageFeatureExtractor',
    'TextFeatureExtractor',
    'MultiModalFusion',
]
