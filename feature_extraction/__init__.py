"""
Feature Extraction Framework
Comprehensive pipelines for CNN and Transformer-based feature extraction
Supports supervised, weakly-supervised, and self-supervised learning
"""

__version__ = "1.0.0"

from .cnn_extractors import (
    BasicCNNExtractor,
    IntermediateCNNExtractor,
    AdvancedCNNExtractor,
    get_cnn_extractor
)

from .transformer_extractors import (
    ImageTransformerExtractor,
    VideoTransformerExtractor,
    get_transformer_extractor
)

from .multi_modal import (
    VideoFeatureExtractor,
    ImageFeatureExtractor,
    TextFeatureExtractor,
    MultiModalFusion
)

from .label_processing import (
    VerbNounProcessor,
    DescriptionGenerator
)

__all__ = [
    'BasicCNNExtractor',
    'IntermediateCNNExtractor',
    'AdvancedCNNExtractor',
    'get_cnn_extractor',
    'ImageTransformerExtractor',
    'VideoTransformerExtractor',
    'get_transformer_extractor',
    'VideoFeatureExtractor',
    'ImageFeatureExtractor',
    'TextFeatureExtractor',
    'MultiModalFusion',
    'VerbNounProcessor',
    'DescriptionGenerator',
]
