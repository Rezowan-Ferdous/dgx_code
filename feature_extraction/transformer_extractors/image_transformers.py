"""
Transformer-based Image Feature Extractors
ViT, DeiT, Swin Transformer, DINO for image feature extraction
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, List, Dict


class ImageTransformerExtractor(nn.Module):
    """
    General image transformer feature extractor

    Supports:
    - Vision Transformer (ViT)
    - Data-efficient Image Transformers (DeiT)
    - Swin Transformer
    - Other transformer variants
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
        extract_features: str = "cls",  # 'cls', 'mean', 'all'
        num_layers_to_extract: Optional[int] = None
    ):
        """
        Initialize image transformer extractor

        Args:
            model_name: Transformer model name (from timm library)
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone weights
            output_dim: Project features to this dimension
            extract_features: How to extract features ('cls', 'mean', 'all')
            num_layers_to_extract: Number of transformer layers to use (None = all)
        """
        super().__init__()

        self.model_name = model_name
        self.extract_features = extract_features

        # Load model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            self.feature_dim = self.backbone.embed_dim
        else:
            # Try to infer from forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.backbone(dummy_input)
                self.feature_dim = dummy_output.shape[-1]

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional projection head
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, output_dim),
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


class ViTExtractor(nn.Module):
    """
    Vision Transformer (ViT) feature extractor with flexible configuration
    """

    def __init__(
        self,
        variant: str = "base",
        patch_size: int = 16,
        img_size: int = 224,
        pretrained: bool = True,
        output_dim: Optional[int] = None,
        extract_intermediate: bool = False
    ):
        """
        Initialize ViT extractor

        Args:
            variant: ViT variant ('tiny', 'small', 'base', 'large', 'huge')
            patch_size: Patch size (16 or 32)
            img_size: Input image size
            pretrained: Use pretrained weights
            output_dim: Output feature dimension
            extract_intermediate: Extract intermediate layer features
        """
        super().__init__()

        # Build model name
        variant_map = {
            'tiny': 'vit_tiny',
            'small': 'vit_small',
            'base': 'vit_base',
            'large': 'vit_large',
            'huge': 'vit_huge'
        }

        model_name = f"{variant_map[variant]}_patch{patch_size}_{img_size}"

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        self.feature_dim = self.backbone.embed_dim
        self.extract_intermediate = extract_intermediate

        # Optional projection
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, output_dim)
            )
            self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        features = self.backbone(x)

        if self.projection is not None:
            features = self.projection(features)

        return features

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features including patch tokens

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary with 'cls_token' and 'patch_tokens'
        """
        # Get patch embeddings
        x = self.backbone.patch_embed(x)

        # Add cls token
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embedding
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        # Pass through transformer blocks
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        # Split cls token and patch tokens
        cls_tokens = x[:, 0]
        patch_tokens = x[:, 1:]

        return {
            'cls_token': cls_tokens,
            'patch_tokens': patch_tokens,
            'all_tokens': x
        }

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim


class DINOExtractor(nn.Module):
    """
    DINO (self-supervised ViT) feature extractor
    Pre-trained with self-distillation
    """

    def __init__(
        self,
        variant: str = "vit_base",
        patch_size: int = 16,
        pretrained: bool = True,
        output_dim: Optional[int] = None,
        use_register_tokens: bool = False
    ):
        """
        Initialize DINO extractor

        Args:
            variant: DINO variant ('vit_small', 'vit_base')
            patch_size: Patch size (8 or 16)
            pretrained: Use DINO pretrained weights
            output_dim: Output feature dimension
            use_register_tokens: Use register tokens (DINOv2)
        """
        super().__init__()

        # Load DINO model
        if pretrained:
            try:
                # Try to load DINOv2 from torch.hub
                if variant == "vit_small" and patch_size == 14:
                    self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                elif variant == "vit_base" and patch_size == 14:
                    self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                elif variant == "vit_large" and patch_size == 14:
                    self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                else:
                    # Fallback to DINO v1
                    self.backbone = torch.hub.load('facebookresearch/dino:main', f'dino_{variant}{patch_size}')
            except:
                # Fallback to timm
                model_name = f"{variant}_patch{patch_size}_224"
                self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            model_name = f"{variant}_patch{patch_size}_224"
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        # Get feature dimension
        if hasattr(self.backbone, 'embed_dim'):
            self.feature_dim = self.backbone.embed_dim
        else:
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


class SwinTransformerExtractor(nn.Module):
    """
    Swin Transformer feature extractor
    Hierarchical vision transformer
    """

    def __init__(
        self,
        variant: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        output_dim: Optional[int] = None,
        extract_hierarchical: bool = False
    ):
        """
        Initialize Swin Transformer extractor

        Args:
            variant: Swin variant
            pretrained: Use pretrained weights
            output_dim: Output feature dimension
            extract_hierarchical: Extract features from all stages
        """
        super().__init__()

        self.extract_hierarchical = extract_hierarchical

        if extract_hierarchical:
            # Load with feature extraction at all stages
            self.backbone = timm.create_model(
                variant,
                pretrained=pretrained,
                features_only=True
            )
            # Get feature dimensions from all stages
            self.stage_dims = [info['num_chs'] for info in self.backbone.feature_info]
            self.feature_dim = sum(self.stage_dims)
        else:
            # Load normally
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
                nn.Linear(self.feature_dim, output_dim)
            )
            self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        if self.extract_hierarchical:
            # Extract from all stages
            stage_features = self.backbone(x)

            # Global average pool each stage
            pooled_features = []
            for feat in stage_features:
                pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                pooled_features.append(pooled.flatten(1))

            # Concatenate all stage features
            features = torch.cat(pooled_features, dim=1)
        else:
            features = self.backbone(x)

        if self.projection is not None:
            features = self.projection(features)

        return features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim
