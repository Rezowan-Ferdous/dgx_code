"""Label processing utilities for verb-noun merging and description generation"""

from .verb_noun_processor import VerbNounProcessor
from .description_generator import DescriptionGenerator
from .label_encoder import LabelEncoder

__all__ = [
    'VerbNounProcessor',
    'DescriptionGenerator',
    'LabelEncoder',
]
