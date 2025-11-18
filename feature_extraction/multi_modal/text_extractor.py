"""
Text feature extractor using transformers
Supports BERT, RoBERTa, CLIP text encoder
"""

import torch
import torch.nn as nn
from typing import Optional, List
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer


class TextFeatureExtractor(nn.Module):
    """
    Text feature extractor using transformer-based models
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: Optional[int] = None,
        max_length: int = 77,
        freeze_backbone: bool = False,
        pooling: str = "cls"  # 'cls', 'mean', 'max'
    ):
        """
        Initialize text feature extractor

        Args:
            model_name: Hugging Face model name or 'clip'
            output_dim: Output feature dimension (None = use model dim)
            max_length: Maximum sequence length
            freeze_backbone: Freeze backbone weights
            pooling: Pooling strategy for token embeddings
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling

        # Load model and tokenizer
        if "clip" in model_name.lower():
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.backbone = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.feature_dim = 512
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            self.feature_dim = self.backbone.config.hidden_size

        # Freeze if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional projection
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
            self.output_dim = output_dim
        else:
            self.output_dim = self.feature_dim

    def tokenize(self, texts: List[str], device: str = "cuda") -> dict:
        """
        Tokenize texts

        Args:
            texts: List of text strings
            device: Device to put tensors on

        Returns:
            Tokenized inputs
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {k: v.to(device) for k, v in inputs.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract text features

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Text features [B, output_dim]
        """
        # Get model outputs
        if "clip" in self.model_name.lower():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # CLIP uses pooler output
            features = outputs.pooler_output
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

            # Apply pooling
            if self.pooling == "cls":
                features = outputs.last_hidden_state[:, 0]  # CLS token
            elif self.pooling == "mean":
                # Mean pooling with attention mask
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                features = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            elif self.pooling == "max":
                features = outputs.last_hidden_state.max(dim=1)[0]
            else:
                raise ValueError(f"Unsupported pooling: {self.pooling}")

        # Apply projection if available
        if self.projection is not None:
            features = self.projection(features)

        return features

    def encode_texts(self, texts: List[str], device: str = "cuda") -> torch.Tensor:
        """
        Convenience method to encode text strings directly

        Args:
            texts: List of text strings
            device: Device to use

        Returns:
            Text features [B, output_dim]
        """
        inputs = self.tokenize(texts, device)
        return self.forward(**inputs)
