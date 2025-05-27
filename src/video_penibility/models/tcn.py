"""TCN model for sequence modeling."""

import torch
import torch.nn as nn
from typing import List
from .base import BaseModel


class TemporalBlock(nn.Module):
    """Temporal block for TCN."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block.

        Args:
            n_inputs: Number of input channels.
            n_outputs: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            dilation: Dilation factor.
            padding: Padding size.
            dropout: Dropout rate.
        """
        super().__init__()

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass through temporal block.

        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            Output tensor of same shape.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the end of sequence."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TCNModel(BaseModel):
    """Temporal Convolutional Network for sequence regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1,
        **kwargs
    ):
        """Initialize TCN model.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension (used to set num_channels if not provided).
            num_channels: List of hidden channel sizes for each layer.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
            output_dim: Output dimension.
            **kwargs: Additional arguments.
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)

        # Store hidden_dim for compatibility
        self.hidden_dim = hidden_dim

        if num_channels is None:
            # Use hidden_dim to create a reasonable channel progression
            num_channels = [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4]

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # TCN expects input in format (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)

        # Pass through TCN layers
        y = self.network(x)  # (batch_size, num_channels[-1], seq_len)

        # Global average pooling
        pooled = self.global_pool(y).squeeze(-1)  # (batch_size, num_channels[-1])

        # Final output projection
        output = self.output_projection(pooled)  # (batch_size, output_dim)

        return output

    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN.

        Returns:
            Receptive field size.
        """
        num_levels = len(self.network)
        receptive_field = 1

        for i in range(num_levels):
            dilation_size = 2**i
            receptive_field += (self.kernel_size - 1) * dilation_size

        return receptive_field
