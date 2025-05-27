"""Model factory for creating models based on configuration."""

from typing import Dict, Any, Type
import logging

from .base import BaseModel
from .rnn import RNNModel, LSTMModel, GRUModel
from .transformer import TransformerModel
from .tcn import TCNModel
from ..config.schema import ModelConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating models based on configuration."""

    _model_registry: Dict[str, Type[BaseModel]] = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "rnn": RNNModel,
        "transformer": TransformerModel,
        "tcn": TCNModel,
    }

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class.

        Args:
            name: Name to register the model under.
            model_class: Model class to register.
        """
        cls._model_registry[name.lower()] = model_class
        logger.info(f"Registered model: {name} -> {model_class.__name__}")

    @classmethod
    def create_model(cls, model_config: ModelConfig, input_dim: int) -> BaseModel:
        """Create a model based on configuration.

        Args:
            model_config: Model configuration.
            input_dim: Input feature dimension.

        Returns:
            Instantiated model.

        Raises:
            ValueError: If model type is not supported.
        """
        model_name = model_config.name.lower()

        if model_name not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"Unsupported model type: {model_name}. "
                f"Available models: {available_models}"
            )

        model_class = cls._model_registry[model_name]

        # Prepare model arguments
        model_args = {
            "input_dim": input_dim,
            "hidden_dim": model_config.hidden_dim,
            "num_layers": model_config.num_layers,
            "output_dim": model_config.output_dim,
            "dropout": model_config.dropout,
        }

        # Add model-specific arguments
        if model_name == "rnn":
            model_args["rnn_type"] = model_config.name.lower()
        elif model_name == "transformer":
            model_args["num_heads"] = model_config.num_heads
        elif model_name == "tcn":
            model_args["kernel_size"] = model_config.kernel_size
            # TCN doesn't use num_layers in the same way - remove it
            del model_args["num_layers"]
        # Note: lstm and gru models set rnn_type internally, so we don't pass it

        # Add any extra parameters
        model_args.update(model_config.extra_params)

        try:
            model = model_class(**model_args)
            logger.info(
                f"Created {model_name} model with {model.get_num_parameters():,} parameters"
            )
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise

    @classmethod
    def list_available_models(cls) -> list:
        """List all available model types.

        Returns:
            List of available model names.
        """
        return list(cls._model_registry.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with model information.
        """
        model_name = model_name.lower()

        if model_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = cls._model_registry[model_name]

        return {
            "name": model_name,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "docstring": model_class.__doc__,
        }
