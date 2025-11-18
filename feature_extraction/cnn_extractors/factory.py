"""
Factory function for creating CNN extractors
"""

from typing import Optional
from .basic import BasicCNNExtractor, SimpleConvNet
from .intermediate import IntermediateCNNExtractor, ResNetFeatureExtractor
from .advanced import AdvancedCNNExtractor, EfficientNetFeatureExtractor, ConvNeXtFeatureExtractor


def get_cnn_extractor(
    level: str = "intermediate",
    backbone: str = "resnet50",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    output_dim: Optional[int] = None,
    **kwargs
):
    """
    Factory function to create CNN feature extractors

    Args:
        level: Complexity level ('basic', 'intermediate', 'advanced')
        backbone: Specific backbone architecture
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone parameters
        output_dim: Output feature dimension
        **kwargs: Additional arguments for specific extractors

    Returns:
        CNN feature extractor module

    Examples:
        >>> # Basic VGG extractor
        >>> extractor = get_cnn_extractor('basic', 'vgg16', pretrained=True)

        >>> # Intermediate ResNet extractor
        >>> extractor = get_cnn_extractor('intermediate', 'resnet50', output_dim=512)

        >>> # Advanced EfficientNet extractor
        >>> extractor = get_cnn_extractor('advanced', 'efficientnet_b3', pretrained=True)
    """

    if level == "basic":
        if backbone in ["vgg16", "vgg19", "alexnet"]:
            return BasicCNNExtractor(
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=output_dim,
                **kwargs
            )
        elif backbone == "simple":
            return SimpleConvNet(output_dim=output_dim or 512, **kwargs)
        else:
            raise ValueError(f"Unsupported basic backbone: {backbone}")

    elif level == "intermediate":
        if "resnet" in backbone:
            if kwargs.get('use_layer_selection', False):
                return ResNetFeatureExtractor(
                    variant=backbone,
                    pretrained=pretrained,
                    **kwargs
                )
            else:
                return IntermediateCNNExtractor(
                    backbone=backbone,
                    pretrained=pretrained,
                    freeze_backbone=freeze_backbone,
                    output_dim=output_dim,
                    **kwargs
                )
        elif backbone in ["densenet121", "densenet161", "mobilenet_v2", "mobilenet_v3_large"]:
            return IntermediateCNNExtractor(
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=output_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported intermediate backbone: {backbone}")

    elif level == "advanced":
        if "efficientnet" in backbone:
            if kwargs.get('use_multi_scale', False):
                return EfficientNetFeatureExtractor(
                    variant=backbone,
                    pretrained=pretrained,
                    output_dim=output_dim,
                    **kwargs
                )
            else:
                return AdvancedCNNExtractor(
                    backbone=backbone,
                    pretrained=pretrained,
                    freeze_backbone=freeze_backbone,
                    output_dim=output_dim,
                    **kwargs
                )
        elif "convnext" in backbone:
            return ConvNeXtFeatureExtractor(
                variant=backbone,
                pretrained=pretrained,
                output_dim=output_dim
            )
        else:
            return AdvancedCNNExtractor(
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=output_dim,
                **kwargs
            )

    else:
        raise ValueError(f"Unsupported level: {level}. Choose from 'basic', 'intermediate', 'advanced'")


# Convenience functions
def get_basic_extractor(backbone: str = "vgg16", **kwargs):
    """Get basic CNN extractor"""
    return get_cnn_extractor("basic", backbone, **kwargs)


def get_intermediate_extractor(backbone: str = "resnet50", **kwargs):
    """Get intermediate CNN extractor"""
    return get_cnn_extractor("intermediate", backbone, **kwargs)


def get_advanced_extractor(backbone: str = "efficientnet_b3", **kwargs):
    """Get advanced CNN extractor"""
    return get_cnn_extractor("advanced", backbone, **kwargs)
