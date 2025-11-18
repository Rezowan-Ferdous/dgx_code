"""
Transformer-based Video Feature Extractors
TimeSformer, VideoMAE, ViViT for video feature extraction
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import math


class VideoTransformerExtractor(nn.Module):
    """
    General video transformer feature extractor

    Supports:
    - Temporal modeling with transformers
    - Spatio-temporal attention
    - Frame-level and video-level features
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: Optional[int] = None,
        temporal_encoding: str = "learned"  # 'learned', 'sinusoidal', 'none'
    ):
        """
        Initialize video transformer extractor

        Args:
            img_size: Input image size
            patch_size: Patch size
            num_frames: Number of frames to process
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            output_dim: Output feature dimension
            temporal_encoding: Type of temporal position encoding
        """
        super().__init__()

        self.num_frames = num_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        # Temporal position embeddings
        if temporal_encoding == "learned":
            self.temporal_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dim)
            )
        elif temporal_encoding == "sinusoidal":
            self.temporal_embed = self._get_sinusoidal_encoding(num_frames, embed_dim)
            self.temporal_embed.requires_grad = False
        else:
            self.temporal_embed = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Optional projection
        self.projection = None
        self.feature_dim = embed_dim
        if output_dim is not None:
            self.projection = nn.Linear(embed_dim, output_dim)
            self.feature_dim = output_dim

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.temporal_embed is not None and self.temporal_embed.requires_grad:
            nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def _get_sinusoidal_encoding(self, num_frames: int, dim: int) -> torch.Tensor:
        """Generate sinusoidal position encoding for temporal dimension"""
        position = torch.arange(num_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        encoding = torch.zeros(num_frames, dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract video features

        Args:
            x: Input video [B, T, C, H, W] or [B, C, T, H, W]

        Returns:
            Video features [B, feature_dim]
        """
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # [B, C, T, H, W]
                B, C, T, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            else:  # [B, T, C, H, W]
                B, T, C, H, W = x.shape
        else:
            raise ValueError(f"Expected 5D input, got {x.dim()}D")

        # Process each frame
        tokens_list = []
        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]
            patches = self.patch_embed(frame)  # [B, embed_dim, H', W']
            patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            tokens_list.append(patches)

        # Stack frame tokens: [B, T, num_patches, embed_dim]
        tokens = torch.stack(tokens_list, dim=1)

        # Reshape for processing: [B, T*num_patches, embed_dim]
        B, T, N, D = tokens.shape
        tokens = tokens.reshape(B, T*N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Add spatial position embeddings (repeated for each frame)
        pos_embed_repeated = self.pos_embed.repeat(1, T, 1)
        tokens[:, 1:] = tokens[:, 1:] + pos_embed_repeated

        # Add temporal embeddings
        if self.temporal_embed is not None:
            # Expand temporal embeddings for all patches in each frame
            temp_embed = self.temporal_embed.unsqueeze(2).repeat(1, 1, N, 1)
            temp_embed = temp_embed.reshape(1, T*N, D)
            tokens[:, 1:] = tokens[:, 1:] + temp_embed

        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Use CLS token as video representation
        video_features = tokens[:, 0]

        # Apply projection if available
        if self.projection is not None:
            video_features = self.projection(video_features)

        return video_features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim


class TransformerBlock(nn.Module):
    """Standard transformer block"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSformerExtractor(nn.Module):
    """
    TimeSformer: Divided Space-Time Attention for video
    Efficient spatio-temporal modeling
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        output_dim: Optional[int] = None
    ):
        """Initialize TimeSformer extractor"""
        super().__init__()

        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        # Divided attention blocks (space then time)
        self.blocks = nn.ModuleList([
            DividedAttentionBlock(embed_dim, num_heads, num_patches, num_frames)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection
        self.projection = None
        self.feature_dim = embed_dim
        if output_dim is not None:
            self.projection = nn.Linear(embed_dim, output_dim)
            self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features with divided space-time attention

        Args:
            x: Input video [B, T, C, H, W]

        Returns:
            Video features [B, feature_dim]
        """
        if x.shape[1] == 3:  # [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        B, T, C, H, W = x.shape

        # Patch embedding for all frames
        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)  # [B*T, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, embed_dim]

        # Reshape to [B, T, num_patches, embed_dim]
        x = x.reshape(B, T, self.num_patches, self.embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, T, -1, -1)  # [B, T, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=2)  # [B, T, num_patches+1, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed.unsqueeze(1)

        # Add temporal embeddings
        x = x + self.temporal_embed.unsqueeze(2)

        # Reshape for divided attention: [B, T*(num_patches+1), embed_dim]
        x = x.reshape(B, T * (self.num_patches + 1), self.embed_dim)

        # Apply divided attention blocks
        for block in self.blocks:
            x = block(x, T, self.num_patches + 1)

        x = self.norm(x)

        # Average CLS tokens across time
        cls_tokens = x.reshape(B, T, self.num_patches + 1, self.embed_dim)[:, :, 0, :]
        video_features = cls_tokens.mean(dim=1)

        if self.projection is not None:
            video_features = self.projection(video_features)

        return video_features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim


class DividedAttentionBlock(nn.Module):
    """Divided space-time attention block"""

    def __init__(self, dim: int, num_heads: int, num_patches: int, num_frames: int):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames

        # Spatial attention
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Temporal attention
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor, T: int, N: int) -> torch.Tensor:
        """
        Args:
            x: Input [B, T*N, D]
            T: Number of frames
            N: Number of patches (including CLS)
        """
        B = x.shape[0]
        D = x.shape[2]

        # Spatial attention (within each frame)
        x_spatial = x.reshape(B * T, N, D)
        x_spatial = x_spatial + self.spatial_attn(
            self.norm1(x_spatial),
            self.norm1(x_spatial),
            self.norm1(x_spatial)
        )[0]
        x = x_spatial.reshape(B, T * N, D)

        # Temporal attention (across frames for each patch position)
        x_temp = x.reshape(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_temp = x_temp + self.temporal_attn(
            self.norm2(x_temp),
            self.norm2(x_temp),
            self.norm2(x_temp)
        )[0]
        x = x_temp.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B, T * N, D)

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x


class VideoMAEExtractor(nn.Module):
    """
    VideoMAE: Masked Autoencoding for video
    Pre-trained with masked reconstruction
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        output_dim: Optional[int] = None,
        tubelet_size: int = 2  # Temporal patch size
    ):
        """Initialize VideoMAE extractor"""
        super().__init__()

        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size

        # 3D patch embedding
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

        num_patches = (num_frames // tubelet_size) * (img_size // patch_size) ** 2

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection
        self.projection = None
        self.feature_dim = embed_dim
        if output_dim is not None:
            self.projection = nn.Linear(embed_dim, output_dim)
            self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features

        Args:
            x: Input video [B, C, T, H, W]

        Returns:
            Video features [B, feature_dim]
        """
        # 3D patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use CLS token
        video_features = x[:, 0]

        if self.projection is not None:
            video_features = self.projection(video_features)

        return video_features

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim
