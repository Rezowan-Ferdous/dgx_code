"""
Basic CNN Feature Extractors
VGG, AlexNet, and simple ConvNets for feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple


class BasicCNNExtractor(nn.Module):
    """
    Basic CNN feature extractor using VGG or AlexNet

    Suitable for:
    - Simple classification tasks
    - Transfer learning baseline
    - Quick prototyping
    """

    def __init__(
        self,
        backbone: str = "vgg16",
        pretrained: bool = True,
        feature_layer: Optional[str] = None,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None
    ):
        """
        Initialize basic CNN extractor

        Args:
            backbone: CNN architecture ('vgg16', 'vgg19', 'alexnet')
            pretrained: Use ImageNet pretrained weights
            feature_layer: Layer to extract features from (None = last layer)
            freeze_backbone: Freeze backbone weights
            output_dim: Project features to this dimension (None = no projection)
        """
        super().__init__()

        self.backbone_name = backbone
        self.feature_layer = feature_layer

        # Load backbone
        if backbone == "vgg16":
            self.backbone = models.vgg16(pretrained=pretrained)
            self.feature_dim = 4096  # FC layer output
        elif backbone == "vgg19":
            self.backbone = models.vgg19(pretrained=pretrained)
            self.feature_dim = 4096
        elif backbone == "alexnet":
            self.backbone = models.alexnet(pretrained=pretrained)
            self.feature_dim = 4096
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classifier if extracting from features
        if feature_layer == "features":
            if "vgg" in backbone:
                self.backbone.classifier = nn.Identity()
                self.feature_dim = 512 * 7 * 7  # VGG feature map
            elif backbone == "alexnet":
                self.backbone.classifier = nn.Identity()
                self.feature_dim = 256 * 6 * 6

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Features [B, feature_dim]
        """
        features = self.backbone(x)

        # Flatten if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # Apply projection if available
        if self.projection is not None:
            features = self.projection(features)

        return features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple intermediate layers

        Args:
            x: Input images [B, C, H, W]
            layer_names: List of layer names to extract from

        Returns:
            Dictionary mapping layer names to features
        """
        features = {}

        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        # Register hooks
        hooks = []
        for name, module in self.backbone.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        _ = self.backbone(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features


class SimpleConvNet(nn.Module):
    """
    Simple ConvNet for custom architectures
    Useful for small datasets or specific tasks
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_conv_layers: int = 4,
        base_channels: int = 64,
        feature_dim: int = 512
    ):
        """
        Initialize simple ConvNet

        Args:
            input_channels: Number of input channels
            num_conv_layers: Number of convolutional layers
            base_channels: Base number of channels (doubles each layer)
            feature_dim: Output feature dimension
        """
        super().__init__()

        # Build convolutional layers
        layers = []
        in_channels = input_channels
        out_channels = base_channels

        for i in range(num_conv_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)

        self.features = nn.Sequential(*layers)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to feature_dim
        self.fc = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim
