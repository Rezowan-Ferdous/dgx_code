"""
Intermediate CNN Feature Extractors
ResNet, DenseNet, and MobileNet for feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional


class IntermediateCNNExtractor(nn.Module):
    """
    Intermediate CNN feature extractor using ResNet, DenseNet, or MobileNet

    Suitable for:
    - Standard computer vision tasks
    - Good accuracy-efficiency tradeoff
    - Most common use case
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        feature_layer: str = "avgpool",
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
        use_multi_scale: bool = False
    ):
        """
        Initialize intermediate CNN extractor

        Args:
            backbone: CNN architecture ('resnet50', 'resnet101', 'densenet121', 'mobilenet_v2')
            pretrained: Use ImageNet pretrained weights
            feature_layer: Layer to extract features from
            freeze_backbone: Freeze backbone weights
            output_dim: Project features to this dimension
            use_multi_scale: Extract multi-scale features
        """
        super().__init__()

        self.backbone_name = backbone
        self.feature_layer = feature_layer
        self.use_multi_scale = use_multi_scale

        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            self.feature_dim = 1024
        elif backbone == "densenet161":
            self.backbone = models.densenet161(pretrained=pretrained)
            self.feature_dim = 2208
        elif backbone == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.feature_dim = 1280
        elif backbone == "mobilenet_v3_large":
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            self.feature_dim = 960
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classification head
        if "resnet" in backbone:
            self.backbone.fc = nn.Identity()
        elif "densenet" in backbone:
            self.backbone.classifier = nn.Identity()
        elif "mobilenet" in backbone:
            self.backbone.classifier = nn.Identity()

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional projection head
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.feature_dim = output_dim

        # Multi-scale feature extraction
        if use_multi_scale:
            self._setup_multi_scale()

    def _setup_multi_scale(self):
        """Setup hooks for multi-scale feature extraction"""
        self.multi_scale_features = {}

        def hook_fn(name):
            def hook(module, input, output):
                self.multi_scale_features[name] = output
            return hook

        if "resnet" in self.backbone_name:
            self.backbone.layer1.register_forward_hook(hook_fn('layer1'))  # 256 channels
            self.backbone.layer2.register_forward_hook(hook_fn('layer2'))  # 512 channels
            self.backbone.layer3.register_forward_hook(hook_fn('layer3'))  # 1024 channels
            self.backbone.layer4.register_forward_hook(hook_fn('layer4'))  # 2048 channels

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

    def forward_multi_scale(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary of features at different scales
        """
        if not self.use_multi_scale:
            raise ValueError("Multi-scale not enabled. Set use_multi_scale=True")

        self.multi_scale_features = {}
        final_features = self.backbone(x)

        # Add final features
        self.multi_scale_features['final'] = final_features

        return self.multi_scale_features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim


class ResNetFeatureExtractor(nn.Module):
    """
    Specialized ResNet feature extractor with flexible layer selection
    """

    def __init__(
        self,
        variant: str = "resnet50",
        pretrained: bool = True,
        output_layers: List[str] = ['layer4'],
        pooling: str = "avg"
    ):
        """
        Initialize ResNet extractor

        Args:
            variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Use ImageNet pretrained weights
            output_layers: Layers to extract features from
            pooling: Pooling type ('avg', 'max', 'none')
        """
        super().__init__()

        # Load ResNet
        if variant == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.layer_dims = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        elif variant == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.layer_dims = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        elif variant == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.layer_dims = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        elif variant == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            self.layer_dims = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        elif variant == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            self.layer_dims = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        self.output_layers = output_layers
        self.pooling = pooling

        # Remove FC layer
        self.backbone.fc = nn.Identity()

        # Setup pooling
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary mapping layer names to features
        """
        features = {}

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1
        x = self.backbone.layer1(x)
        if 'layer1' in self.output_layers:
            if self.pool is not None:
                features['layer1'] = self.pool(x).flatten(1)
            else:
                features['layer1'] = x

        # Layer 2
        x = self.backbone.layer2(x)
        if 'layer2' in self.output_layers:
            if self.pool is not None:
                features['layer2'] = self.pool(x).flatten(1)
            else:
                features['layer2'] = x

        # Layer 3
        x = self.backbone.layer3(x)
        if 'layer3' in self.output_layers:
            if self.pool is not None:
                features['layer3'] = self.pool(x).flatten(1)
            else:
                features['layer3'] = x

        # Layer 4
        x = self.backbone.layer4(x)
        if 'layer4' in self.output_layers:
            if self.pool is not None:
                features['layer4'] = self.pool(x).flatten(1)
            else:
                features['layer4'] = x

        return features

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each layer"""
        return {layer: self.layer_dims[layer] for layer in self.output_layers}
