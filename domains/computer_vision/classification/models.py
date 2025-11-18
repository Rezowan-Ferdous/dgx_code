"""
State-of-the-art Image Classification Models

Supports:
- Vision Transformers (ViT, DeiT, Swin, BEiT)
- CNNs (ResNet, EfficientNet, ConvNeXt, RegNet)
- Hybrid Models (CoAtNet, MaxViT)
- Self-supervised (DINO, MAE, MoCo)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
import timm


class ClassificationModel(nn.Module):
    """Unified interface for image classification models."""

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        global_pool: str = "avg",
        **kwargs
    ):
        """
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate for transformers
            global_pool: Global pooling type ('avg', 'max', 'avgmax')
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Create model using timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
            **kwargs
        )

        # Get model info
        self.model_info = {
            'input_size': self.model.default_cfg.get('input_size', (3, 224, 224)),
            'interpolation': self.model.default_cfg.get('interpolation', 'bilinear'),
            'mean': self.model.default_cfg.get('mean', (0.485, 0.456, 0.406)),
            'std': self.model.default_cfg.get('std', (0.229, 0.224, 0.225)),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        if hasattr(self.model, 'forward_features'):
            return self.model.forward_features(x)
        else:
            # Fallback for models without forward_features
            return self.model.forward_head(x, pre_logits=True)

    def get_classifier(self) -> nn.Module:
        """Get the classification head."""
        return self.model.get_classifier()

    def reset_classifier(self, num_classes: int, global_pool: str = ''):
        """Reset the classification head."""
        self.num_classes = num_classes
        self.model.reset_classifier(num_classes, global_pool)


def get_classification_model(
    architecture: Literal[
        "vit", "deit", "swin", "beit", "convnext", "resnet",
        "efficientnet", "regnet", "dino", "mae", "coatnet", "maxvit"
    ] = "resnet",
    variant: str = "base",
    num_classes: int = 1000,
    pretrained: bool = True,
    img_size: int = 224,
    **kwargs
) -> ClassificationModel:
    """
    Factory function to create classification models.

    Args:
        architecture: Model architecture family
        variant: Model variant (e.g., 'small', 'base', 'large')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        img_size: Input image size

    Returns:
        ClassificationModel instance

    Examples:
        >>> # Vision Transformer
        >>> model = get_classification_model('vit', 'base', num_classes=1000)
        >>>
        >>> # Swin Transformer
        >>> model = get_classification_model('swin', 'base', num_classes=100)
        >>>
        >>> # EfficientNet
        >>> model = get_classification_model('efficientnet', 'b0', num_classes=10)
        >>>
        >>> # DINO (self-supervised)
        >>> model = get_classification_model('dino', 'base', pretrained=True)
    """

    # Map architecture and variant to timm model names
    model_mapping = {
        # Vision Transformers
        "vit": {
            "tiny": "vit_tiny_patch16_224",
            "small": "vit_small_patch16_224",
            "base": "vit_base_patch16_224",
            "large": "vit_large_patch16_224",
            "huge": "vit_huge_patch14_224",
        },
        "deit": {
            "tiny": "deit_tiny_patch16_224",
            "small": "deit_small_patch16_224",
            "base": "deit_base_patch16_224",
        },
        "swin": {
            "tiny": f"swin_tiny_patch4_window7_{img_size}",
            "small": f"swin_small_patch4_window7_{img_size}",
            "base": f"swin_base_patch4_window7_{img_size}",
            "large": f"swin_large_patch4_window7_{img_size}",
        },
        "beit": {
            "base": "beit_base_patch16_224",
            "large": "beit_large_patch16_224",
        },

        # CNNs
        "resnet": {
            "18": "resnet18",
            "34": "resnet34",
            "50": "resnet50",
            "101": "resnet101",
            "152": "resnet152",
            "200": "resnet200",
        },
        "efficientnet": {
            "b0": "efficientnet_b0",
            "b1": "efficientnet_b1",
            "b2": "efficientnet_b2",
            "b3": "efficientnet_b3",
            "b4": "efficientnet_b4",
            "b5": "efficientnet_b5",
            "b6": "efficientnet_b6",
            "b7": "efficientnet_b7",
            "v2_s": "efficientnetv2_rw_s",
            "v2_m": "efficientnetv2_rw_m",
            "v2_l": "efficientnetv2_rw_l",
        },
        "convnext": {
            "tiny": "convnext_tiny",
            "small": "convnext_small",
            "base": "convnext_base",
            "large": "convnext_large",
            "xlarge": "convnext_xlarge",
        },
        "regnet": {
            "y_002": "regnetx_002",
            "y_004": "regnetx_004",
            "y_006": "regnetx_006",
            "y_008": "regnetx_008",
            "y_016": "regnetx_016",
            "y_032": "regnetx_032",
        },

        # Self-supervised
        "dino": {
            "small": "vit_small_patch16_224.dino",
            "base": "vit_base_patch16_224.dino",
        },
        "mae": {
            "base": "vit_base_patch16_224.mae",
            "large": "vit_large_patch16_224.mae",
            "huge": "vit_huge_patch14_224.mae",
        },

        # Hybrid
        "coatnet": {
            "0": "coatnet_0_rw_224",
            "1": "coatnet_1_rw_224",
            "2": "coatnet_2_rw_224",
        },
        "maxvit": {
            "tiny": "maxvit_tiny_tf_224",
            "small": "maxvit_small_tf_224",
            "base": "maxvit_base_tf_224",
            "large": "maxvit_large_tf_224",
        },
    }

    if architecture not in model_mapping:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available: {list(model_mapping.keys())}")

    if variant not in model_mapping[architecture]:
        raise ValueError(f"Unknown variant '{variant}' for {architecture}. "
                        f"Available: {list(model_mapping[architecture].keys())}")

    model_name = model_mapping[architecture][variant]

    # Handle image size variations
    if img_size != 224 and architecture in ["vit", "deit", "beit"]:
        model_name = model_name.replace("224", str(img_size))

    return ClassificationModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


# Model registry for easy access
AVAILABLE_MODELS = {
    "vision_transformers": ["vit", "deit", "swin", "beit"],
    "cnns": ["resnet", "efficientnet", "convnext", "regnet"],
    "self_supervised": ["dino", "mae"],
    "hybrid": ["coatnet", "maxvit"],
}


def list_available_models() -> Dict[str, list]:
    """List all available model architectures grouped by category."""
    return AVAILABLE_MODELS


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    try:
        model = timm.create_model(model_name, pretrained=False)
        return {
            'model_name': model_name,
            'num_params': sum(p.numel() for p in model.parameters()),
            'input_size': model.default_cfg.get('input_size', (3, 224, 224)),
            'pool_size': model.default_cfg.get('pool_size', None),
            'crop_pct': model.default_cfg.get('crop_pct', 0.875),
            'interpolation': model.default_cfg.get('interpolation', 'bilinear'),
            'mean': model.default_cfg.get('mean', (0.485, 0.456, 0.406)),
            'std': model.default_cfg.get('std', (0.229, 0.224, 0.225)),
            'num_classes': model.default_cfg.get('num_classes', 1000),
            'first_conv': model.default_cfg.get('first_conv', 'conv1'),
            'classifier': model.default_cfg.get('classifier', 'fc'),
        }
    except Exception as e:
        return {'error': str(e)}
