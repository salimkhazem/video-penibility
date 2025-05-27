"""RNN-based models (LSTM, GRU) for sequence modeling."""

import torch
import torch.nn as nn
from typing import Literal, Optional

from .base import BaseModel


class RNNModel(BaseModel):
    """RNN-based model for sequence regression.

    Supports LSTM and GRU architectures with configurable layers and dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        output_dim: int = 1,
        rnn_type: Literal["lstm", "gru"] = "gru",
        dropout: float = 0.0,
        bidirectional: bool = False,
        **kwargs,
    ):
        """Initialize RNN model.

        Args:
            input_dim: Number of features in the input embeddings.
            hidden_dim: Number of hidden units in each RNN layer.
            num_layers: Number of RNN layers.
            output_dim: Number of output dimensions (e.g., 1 for regression).
            rnn_type: Type of RNN ('lstm' or 'gru').
            dropout: Dropout rate for regularization (applied between layers).
            bidirectional: Whether to use bidirectional RNN.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Create RNN layer
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(
                f"Unsupported RNN type: {rnn_type}. Choose 'lstm' or 'gru'"
            )

        # Calculate final hidden dimension
        final_hidden_dim = hidden_dim * (2 if bidirectional else 1)

        # Output layer with optional dropout
        layers = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(final_hidden_dim, output_dim))

        self.output_layer = nn.Sequential(*layers)

        # Store configuration
        self._model_config.update(
            {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "rnn_type": rnn_type,
                "dropout": dropout,
                "bidirectional": bidirectional,
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RNN model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Pass through RNN
        # rnn_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        rnn_out, _ = self.rnn(x)

        # Take the last time step's output
        # Shape: (batch_size, hidden_dim * num_directions)
        last_output = rnn_out[:, -1, :]

        # Pass through output layer
        output = self.output_layer(last_output)

        return output

    def get_rnn_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get all RNN outputs (for analysis or attention mechanisms).

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            All RNN outputs of shape (batch_size, sequence_length, hidden_dim * num_directions).
        """
        with torch.no_grad():
            rnn_out, _ = self.rnn(x)
        return rnn_out

    def init_hidden(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> tuple:
        """Initialize hidden state for RNN.

        Args:
            batch_size: Batch size.
            device: Device to create tensors on.

        Returns:
            Tuple of hidden states (h_0, c_0 for LSTM or just h_0 for GRU).
        """
        if device is None:
            device = self.get_device()

        num_directions = 2 if self.bidirectional else 1
        hidden_shape = (self.num_layers * num_directions, batch_size, self.hidden_dim)

        if self.rnn_type == "lstm":
            h_0 = torch.zeros(hidden_shape, device=device)
            c_0 = torch.zeros(hidden_shape, device=device)
            return h_0, c_0
        else:  # GRU
            h_0 = torch.zeros(hidden_shape, device=device)
            return (h_0,)

    def __str__(self) -> str:
        """String representation of the model."""
        return f"""RNNModel(
  {self.rnn_type.upper()}(
    input_size={self.input_dim},
    hidden_size={self.hidden_dim},
    num_layers={self.num_layers},
    bidirectional={self.bidirectional},
    dropout={self.dropout}
  )
  Output: Linear({self.hidden_dim * (2 if self.bidirectional else 1)} -> {self.output_dim})
)"""


class LSTMModel(RNNModel):
    """LSTM model convenience class."""

    def __init__(self, input_dim: int, **kwargs):
        """Initialize LSTM model.

        Args:
            input_dim: Number of input features.
            **kwargs: Additional arguments passed to RNNModel.
        """
        super().__init__(input_dim=input_dim, rnn_type="lstm", **kwargs)


class GRUModel(RNNModel):
    """GRU model convenience class."""

    def __init__(self, input_dim: int, **kwargs):
        """Initialize GRU model.

        Args:
            input_dim: Number of input features.
            **kwargs: Additional arguments passed to RNNModel.
        """
        super().__init__(input_dim=input_dim, rnn_type="gru", **kwargs)
