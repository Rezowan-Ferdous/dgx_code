"""Factory function for creating Transformer extractors"""

from typing import Optional
from .image_transformers import ImageTransformerExtractor, ViTExtractor, DINOExtractor, SwinTransformerExtractor
from .video_transformers import VideoTransformerExtractor, TimeSformerExtractor, VideoMAEExtractor


def get_transformer_extractor(
    modality: str = "image",
    model_type: str = "vit",
    variant: str = "base",
    pretrained: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
):
    """
    Factory function to create Transformer feature extractors

    Args:
        modality: Input modality ('image' or 'video')
        model_type: Model type ('vit', 'dino', 'swin', 'timesformer', 'videomae')
        variant: Model variant
        pretrained: Use pretrained weights
        output_dim: Output feature dimension
        **kwargs: Additional model-specific arguments

    Returns:
        Transformer feature extractor

    Examples:
        >>> # Vision Transformer for images
        >>> extractor = get_transformer_extractor('image', 'vit', 'base')

        >>> # DINO for self-supervised features
        >>> extractor = get_transformer_extractor('image', 'dino', 'vit_base')

        >>> # TimeSformer for videos
        >>> extractor = get_transformer_extractor('video', 'timesformer')
    """

    if modality == "image":
        if model_type == "vit":
            return ViTExtractor(
                variant=variant,
                pretrained=pretrained,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == "dino":
            return DINOExtractor(
                variant=variant,
                pretrained=pretrained,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == "swin":
            variant_name = f"swin_{variant}_patch4_window7_224"
            return SwinTransformerExtractor(
                variant=variant_name,
                pretrained=pretrained,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == "general":
            return ImageTransformerExtractor(
                model_name=variant,
                pretrained=pretrained,
                output_dim=output_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported image model type: {model_type}")

    elif modality == "video":
        if model_type == "timesformer":
            return TimeSformerExtractor(
                pretrained=pretrained,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == "videomae":
            return VideoMAEExtractor(
                pretrained=pretrained,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == "general":
            return VideoTransformerExtractor(
                output_dim=output_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported video model type: {model_type}")

    else:
        raise ValueError(f"Unsupported modality: {modality}")
