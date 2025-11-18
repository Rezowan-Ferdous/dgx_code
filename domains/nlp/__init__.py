"""
Natural Language Processing Module

This module provides state-of-the-art models and datasets for various NLP tasks:
- Text Classification (Sentiment, Topic, etc.)
- Named Entity Recognition (NER)
- Question Answering (QA)
- Text Generation
- Summarization
- Translation
- Sequence Labeling
"""

from .text_classification import (
    TextClassificationTask,
    get_text_classification_model,
    get_text_classification_dataset
)
from .ner import (
    NERTask,
    get_ner_model,
    get_ner_dataset
)
from .qa import (
    QATask,
    get_qa_model,
    get_qa_dataset
)
from .generation import (
    TextGenerationTask,
    get_generation_model,
)

__all__ = [
    # Text Classification
    'TextClassificationTask',
    'get_text_classification_model',
    'get_text_classification_dataset',

    # NER
    'NERTask',
    'get_ner_model',
    'get_ner_dataset',

    # QA
    'QATask',
    'get_qa_model',
    'get_qa_dataset',

    # Generation
    'TextGenerationTask',
    'get_generation_model',
]
