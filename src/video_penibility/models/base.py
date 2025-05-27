"""Base model class providing common functionality."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """Base class for all models in the framework.

    Provides common functionality like parameter counting, device management,
    and standardized interfaces.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, **kwargs):
        """Initialize base model.

        Args:
            input_dim: Dimensionality of input features.
            output_dim: Dimensionality of output (usually 1 for regression).
            **kwargs: Additional model-specific arguments.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._model_config = kwargs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        pass

    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> Dict[str, int]:
        """Get detailed parameter summary.

        Returns:
            Dictionary with parameter statistics.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }

    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze all parameters.

        Args:
            freeze: Whether to freeze (True) or unfreeze (False) parameters.
        """
        for param in self.parameters():
            param.requires_grad = not freeze

        logger.info(f"Parameters {'frozen' if freeze else 'unfrozen'}")

    def get_device(self) -> torch.device:
        """Get the device of the model.

        Returns:
            Device where the model parameters are located.
        """
        return next(self.parameters()).device

    def to_device(self, device: torch.device) -> "BaseModel":
        """Move model to specified device.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        self.to(device)
        return self

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dictionary containing model configuration.
        """
        return {
            "model_class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            **self._model_config,
        }

    def summary(self) -> str:
        """Get a summary string of the model.

        Returns:
            String summary of the model.
        """
        param_summary = self.get_parameter_summary()

        summary = f"""
{self.__class__.__name__} Summary:
{'=' * 50}
Input Dimension: {self.input_dim}
Output Dimension: {self.output_dim}
Total Parameters: {param_summary['total_parameters']:,}
Trainable Parameters: {param_summary['trainable_parameters']:,}
Device: {self.get_device()}

Architecture:
{str(self)}
"""
        return summary.strip()

    def save_checkpoint(
        self, path: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            additional_info: Additional information to save with checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_config(),
            "model_class": self.__class__.__name__,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, **model_kwargs) -> "BaseModel":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file.
            **model_kwargs: Additional arguments for model initialization.

        Returns:
            Loaded model instance.
        """
        # NOTE: This torch.load is used for trusted model checkpoints only
        # Checkpoints are saved by our own training pipeline
        checkpoint = torch.load(path, map_location="cpu")  # nosec B614

        # Extract model configuration
        model_config = checkpoint.get("model_config", {})
        model_config.update(model_kwargs)

        # Create model instance
        model = cls(**model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model loaded from checkpoint: {path}")
        return model
