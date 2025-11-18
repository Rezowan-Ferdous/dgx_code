"""
Unified video feature extractor supporting both CNN and Transformer backbones
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class VideoFeatureExtractor(nn.Module):
    """
    Unified video feature extractor

    Supports:
    - CNN-based: 2D CNN + temporal pooling, 3D CNN, (2+1)D CNN
    - Transformer-based: TimeSformer, VideoMAE, ViViT
    - Hybrid: CNN features + Temporal Transformer
    """

    def __init__(
        self,
        backbone_type: Literal["cnn", "transformer", "hybrid"] = "hybrid",
        cnn_backbone: Optional[str] = "resnet50",
        transformer_backbone: Optional[str] = "timesformer",
        num_frames: int = 16,
        temporal_pooling: str = "avg",  # 'avg', 'max', 'attention'
        output_dim: int = 512,
        freeze_backbone: bool = False
    ):
        """
        Initialize video feature extractor

        Args:
            backbone_type: Type of backbone ('cnn', 'transformer', 'hybrid')
            cnn_backbone: CNN backbone if using CNN or hybrid
            transformer_backbone: Transformer backbone if using transformer
            num_frames: Number of frames to sample
            temporal_pooling: Temporal pooling method
            output_dim: Output feature dimension
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.backbone_type = backbone_type
        self.num_frames = num_frames
        self.temporal_pooling = temporal_pooling

        if backbone_type == "cnn":
            self._build_cnn_backbone(cnn_backbone, freeze_backbone)
        elif backbone_type == "transformer":
            self._build_transformer_backbone(transformer_backbone, freeze_backbone)
        elif backbone_type == "hybrid":
            self._build_hybrid_backbone(cnn_backbone, freeze_backbone)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # Temporal pooling module
        if temporal_pooling == "attention":
            self.temporal_attention = TemporalAttentionPooling(self.feature_dim)

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        self.output_dim = output_dim

    def _build_cnn_backbone(self, backbone_name: str, freeze: bool):
        """Build CNN backbone for frame-level features"""
        from ..cnn_extractors import get_cnn_extractor

        self.frame_encoder = get_cnn_extractor(
            level="intermediate",
            backbone=backbone_name,
            pretrained=True,
            freeze_backbone=freeze
        )
        self.feature_dim = self.frame_encoder.get_feature_dim()

    def _build_transformer_backbone(self, backbone_name: str, freeze: bool):
        """Build transformer backbone for video"""
        from ..transformer_extractors import get_transformer_extractor

        self.video_encoder = get_transformer_extractor(
            modality="video",
            model_type=backbone_name,
            pretrained=True
        )
        self.feature_dim = self.video_encoder.get_feature_dim()

        if freeze:
            for param in self.video_encoder.parameters():
                param.requires_grad = False

    def _build_hybrid_backbone(self, cnn_backbone: str, freeze: bool):
        """Build hybrid CNN + Transformer backbone"""
        from ..cnn_extractors import get_cnn_extractor

        # Frame-level CNN
        self.frame_encoder = get_cnn_extractor(
            level="intermediate",
            backbone=cnn_backbone,
            pretrained=True,
            freeze_backbone=freeze
        )
        frame_dim = self.frame_encoder.get_feature_dim()

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            dim=frame_dim,
            depth=4,
            num_heads=8
        )
        self.feature_dim = frame_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract video features

        Args:
            x: Input video [B, T, C, H, W] or [B, C, T, H, W]

        Returns:
            Video features [B, output_dim]
        """
        if self.backbone_type == "cnn" or self.backbone_type == "hybrid":
            # Handle input format
            if x.dim() == 5 and x.shape[1] == 3:  # [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

            B, T, C, H, W = x.shape

            # Extract frame-level features
            x_frames = x.reshape(B * T, C, H, W)
            frame_features = self.frame_encoder(x_frames)  # [B*T, D]
            frame_features = frame_features.reshape(B, T, -1)  # [B, T, D]

            if self.backbone_type == "hybrid":
                # Apply temporal transformer
                video_features = self.temporal_transformer(frame_features)
            else:
                # Temporal pooling
                if self.temporal_pooling == "avg":
                    video_features = frame_features.mean(dim=1)
                elif self.temporal_pooling == "max":
                    video_features = frame_features.max(dim=1)[0]
                elif self.temporal_pooling == "attention":
                    video_features = self.temporal_attention(frame_features)
                else:
                    raise ValueError(f"Unsupported pooling: {self.temporal_pooling}")

        else:  # transformer
            # Transformer backbone handles video directly
            video_features = self.video_encoder(x)

        # Project to output dimension
        output = self.projection(video_features)

        return output

    def get_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frame-level features without temporal pooling

        Args:
            x: Input video [B, T, C, H, W]

        Returns:
            Frame features [B, T, feature_dim]
        """
        if self.backbone_type == "transformer":
            raise NotImplementedError("Frame features not available for pure transformer backbone")

        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        B, T, C, H, W = x.shape
        x_frames = x.reshape(B * T, C, H, W)
        frame_features = self.frame_encoder(x_frames)
        frame_features = frame_features.reshape(B, T, -1)

        return frame_features


class TemporalTransformer(nn.Module):
    """Temporal transformer for aggregating frame features"""

    def __init__(self, dim: int, depth: int = 4, num_heads: int = 8):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, dim))  # Learnable for variable length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Frame features [B, T, D]

        Returns:
            Video features [B, D]
        """
        B, T, D = x.shape

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        pos_embed = torch.arange(T + 1, device=x.device).unsqueeze(0).unsqueeze(2)
        pos_embed = pos_embed * self.pos_embed
        x = x + pos_embed

        # Apply transformer
        x = self.transformer(x)

        # Return CLS token
        return x[:, 0]


class TemporalAttentionPooling(nn.Module):
    """Attention-based temporal pooling"""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [B, T, D]

        Returns:
            Pooled features [B, D]
        """
        # Compute attention weights
        weights = self.attention(x)  # [B, T, 1]

        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # [B, D]

        return pooled
