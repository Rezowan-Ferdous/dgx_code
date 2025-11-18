"""Unified image feature extractor"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class ImageFeatureExtractor(nn.Module):
    """
    Unified image feature extractor supporting both CNN and Transformer backbones
    """

    def __init__(
        self,
        backbone_type: Literal["cnn", "transformer"] = "cnn",
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_backbone: bool = False
    ):
        """
        Initialize image feature extractor

        Args:
            backbone_type: 'cnn' or 'transformer'
            backbone_name: Specific backbone model
            pretrained: Use pretrained weights
            output_dim: Output feature dimension
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.backbone_type = backbone_type

        if backbone_type == "cnn":
            from ..cnn_extractors import get_cnn_extractor
            self.backbone = get_cnn_extractor(
                level="intermediate",
                backbone=backbone_name,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=output_dim
            )
        elif backbone_type == "transformer":
            from ..transformer_extractors import get_transformer_extractor
            self.backbone = get_transformer_extractor(
                modality="image",
                model_type="general",
                variant=backbone_name,
                pretrained=pretrained,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract image features

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Image features [B, output_dim]
        """
        return self.backbone(x)
