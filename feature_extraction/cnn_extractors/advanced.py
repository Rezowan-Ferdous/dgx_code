"""
Advanced CNN Feature Extractors
EfficientNet, ConvNeXt, RegNet for state-of-the-art feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional
import timm  # PyTorch Image Models for advanced architectures


class AdvancedCNNExtractor(nn.Module):
    """
    Advanced CNN feature extractor using EfficientNet, ConvNeXt, or RegNet

    Suitable for:
    - State-of-the-art performance
    - Efficient computation
    - Production deployments
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
        drop_path_rate: float = 0.0
    ):
        """
        Initialize advanced CNN extractor

        Args:
            backbone: CNN architecture (efficientnet_b0-b7, convnext_tiny/small/base, regnet_y_400mf)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights
            output_dim: Project features to this dimension
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()

        self.backbone_name = backbone

        # Load backbone using timm
        try:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                drop_path_rate=drop_path_rate
            )
            self.feature_dim = self.backbone.num_features
        except Exception as e:
            # Fallback to torchvision
            if "efficientnet" in backbone:
                if backbone == "efficientnet_b0":
                    self.backbone = models.efficientnet_b0(pretrained=pretrained)
                    self.feature_dim = 1280
                elif backbone == "efficientnet_b1":
                    self.backbone = models.efficientnet_b1(pretrained=pretrained)
                    self.feature_dim = 1280
                elif backbone == "efficientnet_b2":
                    self.backbone = models.efficientnet_b2(pretrained=pretrained)
                    self.feature_dim = 1408
                elif backbone == "efficientnet_b3":
                    self.backbone = models.efficientnet_b3(pretrained=pretrained)
                    self.feature_dim = 1536
                elif backbone == "efficientnet_b4":
                    self.backbone = models.efficientnet_b4(pretrained=pretrained)
                    self.feature_dim = 1792
                elif backbone == "efficientnet_b5":
                    self.backbone = models.efficientnet_b5(pretrained=pretrained)
                    self.feature_dim = 2048
                elif backbone == "efficientnet_b6":
                    self.backbone = models.efficientnet_b6(pretrained=pretrained)
                    self.feature_dim = 2304
                elif backbone == "efficientnet_b7":
                    self.backbone = models.efficientnet_b7(pretrained=pretrained)
                    self.feature_dim = 2560
                else:
                    raise ValueError(f"Unsupported backbone: {backbone}")

                self.backbone.classifier = nn.Identity()
            elif "convnext" in backbone:
                if backbone == "convnext_tiny":
                    self.backbone = models.convnext_tiny(pretrained=pretrained)
                    self.feature_dim = 768
                elif backbone == "convnext_small":
                    self.backbone = models.convnext_small(pretrained=pretrained)
                    self.feature_dim = 768
                elif backbone == "convnext_base":
                    self.backbone = models.convnext_base(pretrained=pretrained)
                    self.feature_dim = 1024
                elif backbone == "convnext_large":
                    self.backbone = models.convnext_large(pretrained=pretrained)
                    self.feature_dim = 1536
                else:
                    raise ValueError(f"Unsupported backbone: {backbone}")

                self.backbone.classifier = nn.Identity()
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional projection head
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Features [B, feature_dim]
        """
        features = self.backbone(x)

        # Apply projection if available
        if self.projection is not None:
            features = self.projection(features)

        return features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim


class EfficientNetFeatureExtractor(nn.Module):
    """
    Specialized EfficientNet feature extractor with multi-scale support
    """

    def __init__(
        self,
        variant: str = "efficientnet_b3",
        pretrained: bool = True,
        extract_stages: List[int] = [4, 6, 8],  # Which stages to extract from
        output_dim: Optional[int] = None
    ):
        """
        Initialize EfficientNet extractor

        Args:
            variant: EfficientNet variant (b0-b7)
            pretrained: Use ImageNet pretrained weights
            extract_stages: Stages to extract features from
            output_dim: Final output dimension
        """
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=extract_stages
        )

        self.extract_stages = extract_stages
        self.feature_info = self.backbone.feature_info

        # Calculate total feature dimension
        self.stage_dims = [info['num_chs'] for info in self.feature_info]
        self.total_dim = sum(self.stage_dims)

        # Optional projection
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.total_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            self.feature_dim = output_dim
        else:
            self.feature_dim = self.total_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features and concatenate

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Concatenated features [B, feature_dim]
        """
        # Extract features from multiple stages
        features_list = self.backbone(x)

        # Global average pooling for each stage
        pooled_features = []
        for feat in features_list:
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate all features
        combined = torch.cat(pooled_features, dim=1)

        # Apply projection if available
        if self.projection is not None:
            combined = self.projection(combined)

        return combined

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim

    def get_stage_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get features from individual stages

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary mapping stage names to features
        """
        features_list = self.backbone(x)

        stage_features = {}
        for idx, (stage_idx, feat) in enumerate(zip(self.extract_stages, features_list)):
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            pooled = pooled.flatten(1)
            stage_features[f'stage_{stage_idx}'] = pooled

        return stage_features


class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt-based feature extractor with hierarchical features
    """

    def __init__(
        self,
        variant: str = "convnext_tiny",
        pretrained: bool = True,
        output_dim: Optional[int] = None
    ):
        """
        Initialize ConvNeXt extractor

        Args:
            variant: ConvNeXt variant (tiny, small, base, large)
            pretrained: Use ImageNet pretrained weights
            output_dim: Final output dimension
        """
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0
        )

        self.feature_dim = self.backbone.num_features

        # Optional projection
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, output_dim),
                nn.GELU()
            )
            self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        features = self.backbone(x)

        if self.projection is not None:
            features = self.projection(features)

        return features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim
