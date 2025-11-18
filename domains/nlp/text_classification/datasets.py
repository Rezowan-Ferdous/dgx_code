"""
Text Classification Datasets

Popular datasets for text classification tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
from datasets import load_dataset
import numpy as np


class TextClassificationDataset(Dataset):
    """Text classification dataset."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def get_text_classification_dataset(
    dataset_name: str = "imdb",
    split: str = "train",
    tokenizer=None,
    max_length: int = 512,
    **kwargs
):
    """
    Factory function for text classification datasets.

    Args:
        dataset_name: Dataset name ('imdb', 'sst2', 'ag_news', 'yelp', 'amazon')
        split: Dataset split
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length

    Returns:
        TextClassificationDataset instance

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> # IMDb sentiment dataset
        >>> dataset = get_text_classification_dataset('imdb', 'train', tokenizer)
        >>>
        >>> # AG News topic classification
        >>> dataset = get_text_classification_dataset('ag_news', 'train', tokenizer)
    """

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load dataset from HuggingFace
    if dataset_name == "imdb":
        data = load_dataset("imdb", split=split)
        texts = data['text']
        labels = data['label']

    elif dataset_name == "sst2":
        data = load_dataset("glue", "sst2", split=split)
        texts = data['sentence']
        labels = data['label']

    elif dataset_name == "ag_news":
        data = load_dataset("ag_news", split=split)
        texts = data['text']
        labels = data['label']

    elif dataset_name == "yelp":
        data = load_dataset("yelp_polarity", split=split)
        texts = data['text']
        labels = data['label']

    elif dataset_name == "amazon":
        data = load_dataset("amazon_polarity", split=split)
        texts = [f"{title} {content}" for title, content in zip(data['title'], data['content'])]
        labels = data['label']

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return TextClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )


# Dataset information
DATASET_INFO = {
    "imdb": {
        "num_classes": 2,
        "task": "sentiment",
        "num_train": 25000,
        "num_test": 25000,
    },
    "sst2": {
        "num_classes": 2,
        "task": "sentiment",
        "num_train": 67349,
        "num_val": 872,
    },
    "ag_news": {
        "num_classes": 4,
        "task": "topic",
        "num_train": 120000,
        "num_test": 7600,
    },
    "yelp": {
        "num_classes": 2,
        "task": "sentiment",
        "num_train": 560000,
        "num_test": 38000,
    },
}
