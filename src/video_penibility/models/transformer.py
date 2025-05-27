"""Transformer model for sequence modeling."""

import torch
import torch.nn as nn
import math
from typing import Optional
from .base import BaseModel


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:, : x.size(1)]


class TransformerModel(BaseModel):
    """Transformer-based model for sequence regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 1000,
        **kwargs
    ):
        """Initialize transformer model.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension (d_model).
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            output_dim: Output dimension.
            dropout: Dropout rate.
            max_seq_length: Maximum sequence length.
            **kwargs: Additional arguments.
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask for variable length sequences.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Boolean mask where True indicates padding positions.
        """
        # Assume padding is done with zeros across all features
        # This is a simple heuristic - could be improved
        mask = x.sum(dim=-1) == 0  # Shape: (batch_size, seq_len)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout_layer(x)

        # Create padding mask
        src_key_padding_mask = self._create_padding_mask(x)

        # Pass through transformer encoder
        encoded = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, hidden_dim)

        # Apply layer normalization
        encoded = self.layer_norm(encoded)

        # Global average pooling over sequence dimension
        # Mask out padding positions
        if src_key_padding_mask.any():
            # Create weights for averaging (ignore padding positions)
            weights = (
                (~src_key_padding_mask).float().unsqueeze(-1)
            )  # (batch_size, seq_len, 1)
            weighted_sum = (encoded * weights).sum(dim=1)  # (batch_size, hidden_dim)
            total_weight = weights.sum(dim=1)  # (batch_size, 1)
            pooled = weighted_sum / (total_weight + 1e-8)  # Avoid division by zero
        else:
            pooled = encoded.mean(dim=1)  # (batch_size, hidden_dim)

        # Final output projection
        output = self.output_projection(pooled)  # (batch_size, output_dim)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights from the last layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Attention weights tensor.
        """
        # This would require modifying the transformer to return attention weights
        # For now, just return a placeholder
        batch_size, seq_len, _ = x.shape
        return torch.ones(batch_size, self.num_heads, seq_len, seq_len)
