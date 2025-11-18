"""
State-of-the-art Text Classification Models

Supports:
- Transformer-based: BERT, RoBERTa, DeBERTa, ELECTRA
- Large Language Models: GPT-2, GPT-Neo
- Lightweight: DistilBERT, ALBERT, MobileBERT
- Domain-specific: BioBERT, SciBERT, FinBERT
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertModel,
    RobertaModel,
    DebertaV2Model,
    ElectraModel,
    DistilBertModel,
    AlbertModel,
)


class TextClassificationModel(nn.Module):
    """Unified interface for text classification models."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.1,
        use_pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        """
        Args:
            model_name: Pretrained model name from HuggingFace
            num_classes: Number of output classes
            dropout: Dropout rate
            use_pretrained: Whether to use pretrained weights
            freeze_encoder: Whether to freeze encoder layers
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained model
        if use_pretrained:
            try:
                # Try to load a model specifically for sequence classification
                self.encoder = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                self.use_auto_model = True
            except:
                # Fallback to base model + custom classifier
                self.encoder = AutoModel.from_pretrained(model_name)
                self.use_auto_model = False
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)
            self.use_auto_model = False

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Add custom classification head if needed
        if not self.use_auto_model:
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_classes)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]

        Returns:
            Logits [batch, num_classes]
        """
        if self.use_auto_model:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            return outputs.logits
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Use [CLS] token representation
            pooled_output = outputs.last_hidden_state[:, 0, :]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            return logits

    def encode_texts(
        self,
        texts: list,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode texts using tokenizer.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return tensor format

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )


def get_text_classification_model(
    architecture: Literal[
        "bert", "roberta", "deberta", "electra", "distilbert",
        "albert", "gpt2", "biobert", "scibert", "finbert"
    ] = "bert",
    variant: str = "base",
    num_classes: int = 2,
    use_pretrained: bool = True,
    **kwargs
) -> TextClassificationModel:
    """
    Factory function to create text classification models.

    Args:
        architecture: Model architecture family
        variant: Model variant ('base', 'large', etc.)
        num_classes: Number of output classes
        use_pretrained: Whether to use pretrained weights

    Returns:
        TextClassificationModel instance

    Examples:
        >>> # BERT for binary classification
        >>> model = get_text_classification_model('bert', 'base', num_classes=2)
        >>>
        >>> # RoBERTa for multi-class classification
        >>> model = get_text_classification_model('roberta', 'large', num_classes=10)
        >>>
        >>> # DeBERTa v3 for sentiment analysis
        >>> model = get_text_classification_model('deberta', 'v3-base', num_classes=3)
        >>>
        >>> # Domain-specific: BioBERT for biomedical text
        >>> model = get_text_classification_model('biobert', 'base', num_classes=5)
    """

    # Map architecture and variant to HuggingFace model names
    model_mapping = {
        "bert": {
            "tiny": "prajjwal1/bert-tiny",
            "mini": "prajjwal1/bert-mini",
            "small": "prajjwal1/bert-small",
            "medium": "prajjwal1/bert-medium",
            "base": "bert-base-uncased",
            "large": "bert-large-uncased",
            "base-cased": "bert-base-cased",
            "large-cased": "bert-large-cased",
        },
        "roberta": {
            "base": "roberta-base",
            "large": "roberta-large",
        },
        "deberta": {
            "base": "microsoft/deberta-base",
            "large": "microsoft/deberta-large",
            "xlarge": "microsoft/deberta-xlarge",
            "v2-xlarge": "microsoft/deberta-v2-xlarge",
            "v2-xxlarge": "microsoft/deberta-v2-xxlarge",
            "v3-base": "microsoft/deberta-v3-base",
            "v3-large": "microsoft/deberta-v3-large",
        },
        "electra": {
            "small": "google/electra-small-discriminator",
            "base": "google/electra-base-discriminator",
            "large": "google/electra-large-discriminator",
        },
        "distilbert": {
            "base": "distilbert-base-uncased",
            "base-cased": "distilbert-base-cased",
        },
        "albert": {
            "base": "albert-base-v2",
            "large": "albert-large-v2",
            "xlarge": "albert-xlarge-v2",
            "xxlarge": "albert-xxlarge-v2",
        },
        "gpt2": {
            "base": "gpt2",
            "medium": "gpt2-medium",
            "large": "gpt2-large",
            "xl": "gpt2-xl",
        },
        # Domain-specific models
        "biobert": {
            "base": "dmis-lab/biobert-base-cased-v1.2",
            "large": "dmis-lab/biobert-large-cased-v1.1",
        },
        "scibert": {
            "base": "allenai/scibert_scivocab_uncased",
            "cased": "allenai/scibert_scivocab_cased",
        },
        "finbert": {
            "base": "ProsusAI/finbert",
        },
    }

    if architecture not in model_mapping:
        raise ValueError(f"Unknown architecture: {architecture}")

    if variant not in model_mapping[architecture]:
        raise ValueError(f"Unknown variant '{variant}' for {architecture}")

    model_name = model_mapping[architecture][variant]

    return TextClassificationModel(
        model_name=model_name,
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        **kwargs
    )


# Model registry
AVAILABLE_MODELS = {
    "general": ["bert", "roberta", "deberta", "electra"],
    "lightweight": ["distilbert", "albert"],
    "generative": ["gpt2"],
    "domain_specific": ["biobert", "scibert", "finbert"],
}


def list_available_models() -> Dict[str, list]:
    """List all available model architectures."""
    return AVAILABLE_MODELS


# Model information
MODEL_INFO = {
    "bert-base": {
        "params": "110M",
        "layers": 12,
        "hidden_size": 768,
        "description": "Original BERT base model",
    },
    "roberta-large": {
        "params": "355M",
        "layers": 24,
        "hidden_size": 1024,
        "description": "RoBERTa with improved training",
    },
    "deberta-v3-large": {
        "params": "304M",
        "layers": 24,
        "hidden_size": 1024,
        "description": "DeBERTa v3 with disentangled attention",
    },
    "distilbert-base": {
        "params": "66M",
        "layers": 6,
        "hidden_size": 768,
        "description": "Distilled BERT (40% smaller, 60% faster)",
    },
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model."""
    return MODEL_INFO.get(model_name, {})
