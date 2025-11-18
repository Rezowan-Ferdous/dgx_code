"""
Multi-modal fusion strategies
Combine features from video, image, and text modalities
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module

    Supports various fusion strategies:
    - Concatenation
    - Attention-based fusion
    - Gated fusion
    - Transformer fusion
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        fusion_method: str = "concat",  # 'concat', 'attention', 'gated', 'transformer'
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize multi-modal fusion

        Args:
            input_dims: Dictionary mapping modality names to feature dimensions
            output_dim: Output fused feature dimension
            fusion_method: Fusion strategy
            hidden_dim: Hidden dimension for attention/gated fusion
        """
        super().__init__()

        self.modalities = list(input_dims.keys())
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_method = fusion_method

        if fusion_method == "concat":
            total_dim = sum(input_dims.values())
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )

        elif fusion_method == "attention":
            hidden_dim = hidden_dim or output_dim
            self.projections = nn.ModuleDict({
                modality: nn.Linear(dim, hidden_dim)
                for modality, dim in input_dims.items()
            })
            self.attention = CrossModalAttention(hidden_dim, len(input_dims))
            self.fusion = nn.Linear(hidden_dim, output_dim)

        elif fusion_method == "gated":
            self.projections = nn.ModuleDict({
                modality: nn.Linear(dim, output_dim)
                for modality, dim in input_dims.items()
            })
            self.gates = nn.ModuleDict({
                modality: nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.Sigmoid()
                )
                for modality, dim in input_dims.items()
            })

        elif fusion_method == "transformer":
            hidden_dim = hidden_dim or output_dim
            self.projections = nn.ModuleDict({
                modality: nn.Linear(dim, hidden_dim)
                for modality, dim in input_dims.items()
            })
            self.transformer_fusion = TransformerFusion(hidden_dim, output_dim)

        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-modal features

        Args:
            features: Dictionary mapping modality names to features [B, D_modality]

        Returns:
            Fused features [B, output_dim]
        """
        if self.fusion_method == "concat":
            # Concatenate all features
            feat_list = [features[mod] for mod in self.modalities if mod in features]
            concatenated = torch.cat(feat_list, dim=1)
            fused = self.fusion(concatenated)

        elif self.fusion_method == "attention":
            # Project all features
            projected = {mod: self.projections[mod](features[mod]) for mod in features}

            # Stack for attention
            feat_stack = torch.stack([projected[mod] for mod in self.modalities if mod in projected], dim=1)

            # Apply attention
            attended = self.attention(feat_stack)

            # Final projection
            fused = self.fusion(attended)

        elif self.fusion_method == "gated":
            # Gated fusion
            gated_features = []
            for mod in self.modalities:
                if mod in features:
                    projected = self.projections[mod](features[mod])
                    gate = self.gates[mod](features[mod])
                    gated_features.append(projected * gate)

            # Sum gated features
            fused = torch.stack(gated_features, dim=0).sum(dim=0)

        elif self.fusion_method == "transformer":
            # Project features
            projected = {mod: self.projections[mod](features[mod]) for mod in features}

            # Stack features
            feat_stack = torch.stack([projected[mod] for mod in self.modalities if mod in projected], dim=1)

            # Transformer fusion
            fused = self.transformer_fusion(feat_stack)

        return fused


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""

    def __init__(self, dim: int, num_modalities: int):
        super().__init__()

        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stacked features [B, num_modalities, D]

        Returns:
            Attended features [B, D]
        """
        # Self-attention across modalities
        attended, _ = self.attention(x, x, x)

        # Residual connection
        x = x + attended
        x = self.norm(x)

        # Average across modalities
        return x.mean(dim=1)


class TransformerFusion(nn.Module):
    """Transformer-based fusion"""

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()

        # Learnable fusion token
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stacked features [B, num_modalities, D]

        Returns:
            Fused features [B, output_dim]
        """
        B = x.shape[0]

        # Add fusion token
        fusion_tokens = self.fusion_token.expand(B, -1, -1)
        x = torch.cat([fusion_tokens, x], dim=1)

        # Apply transformer
        x = self.transformer(x)

        # Use fusion token as output
        fused = x[:, 0]

        # Project to output dimension
        return self.output_proj(fused)
