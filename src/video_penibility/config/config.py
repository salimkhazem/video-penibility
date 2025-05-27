"""Configuration loading and management."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict, fields
import logging

from .schema import (
    Config as ConfigSchema,
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    CrossValidationConfig,
    VisualizationConfig,
)

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for loading and handling experiment configs."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.

        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self._config = ConfigSchema()

        if config_path is not None:
            self.load_from_file(config_path)

        self._config.validate()

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        self.load_from_dict(config_dict)

    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration.
        """
        # Update each section
        if "experiment" in config_dict:
            self._update_dataclass(self._config.experiment, config_dict["experiment"])

        if "data" in config_dict:
            self._update_dataclass(self._config.data, config_dict["data"])

        if "model" in config_dict:
            self._update_dataclass(self._config.model, config_dict["model"])

        if "training" in config_dict:
            self._update_dataclass(self._config.training, config_dict["training"])

        if "cross_validation" in config_dict:
            self._update_dataclass(
                self._config.cross_validation, config_dict["cross_validation"]
            )

        if "visualization" in config_dict:
            self._update_dataclass(
                self._config.visualization, config_dict["visualization"]
            )

    def _update_dataclass(self, obj: Any, updates: Dict[str, Any]) -> None:
        """Update dataclass object with dictionary values.

        Args:
            obj: Dataclass object to update.
            updates: Dictionary with updates.
        """
        field_names = {f.name for f in fields(obj)}

        for key, value in updates.items():
            if key in field_names:
                setattr(obj, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file.

        Args:
            config_path: Path to save configuration file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self._config)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to: {config_path}")

    def merge(self, other_config: Union["Config", Dict[str, Any]]) -> "Config":
        """Merge with another configuration.

        Args:
            other_config: Another Config object or dictionary to merge.

        Returns:
            New Config object with merged settings.
        """
        new_config = Config()
        new_config._config = ConfigSchema(
            experiment=self._config.experiment,
            data=self._config.data,
            model=self._config.model,
            training=self._config.training,
            cross_validation=self._config.cross_validation,
            visualization=self._config.visualization,
        )

        if isinstance(other_config, Config):
            other_dict = asdict(other_config._config)
        else:
            other_dict = other_config

        new_config.load_from_dict(other_dict)
        return new_config

    @property
    def experiment(self) -> ExperimentConfig:
        """Get experiment configuration."""
        return self._config.experiment

    @property
    def data(self) -> DataConfig:
        """Get data configuration."""
        return self._config.data

    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return self._config.model

    @property
    def training(self) -> TrainingConfig:
        """Get training configuration."""
        return self._config.training

    @property
    def cross_validation(self) -> CrossValidationConfig:
        """Get cross-validation configuration."""
        return self._config.cross_validation

    @property
    def visualization(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self._config.visualization

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return asdict(self._config)

    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        """Repr of configuration."""
        return f"Config(experiment='{self.experiment.name}')"
