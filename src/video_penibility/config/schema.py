"""Configuration schema definitions using dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any, Dict
from pathlib import Path
import yaml  # type: ignore


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    name: str = "default_experiment"
    seed: int = 42
    output_dir: str = "results"
    device: str = "auto"  # auto, cuda:0, cpu
    log_level: str = "INFO"
    tags: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Data configuration."""

    annotation_path: str = "data/labels/annotation_file.csv"
    features_type: Literal[
        "facenet",
        "resnet",
        "facemesh",
        "facemesh_dist_v1",
        "facemesh_dist_v2",
        "i3d",
        "swin3d_t",
        "swin3d_s",
        "swin3d_b",
        "libreface",
    ] = "i3d"
    libreface_features: Literal["headpose", "landmarks", "AUs", "all"] = "all"
    data_type: Literal["full_body", "face_1.7"] = "full_body"
    target_normalization: bool = False
    max_timesteps: Optional[int] = None
    features_root: str = "data/features"


@dataclass
class ModelConfig:
    """Model configuration."""

    name: Literal["transformer", "lstm", "gru", "tcn"] = "gru"
    hidden_dim: int = 512
    num_heads: int = 8  # For transformer
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.0
    # TCN specific
    kernel_size: int = 3
    # Additional model parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    early_stopping_patience: int = 30
    optimizer: str = "adamw"
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # Loss and metrics
    loss_function: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mse", "mae", "r2", "ccc"])

    # Checkpointing
    save_best_model: bool = True
    save_last_model: bool = False
    save_checkpoints: bool = False
    checkpoint_frequency: int = 10


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration."""

    n_splits: int = 5
    strategy: Literal["subject_wise", "random"] = "subject_wise"
    shuffle: bool = True
    random_state: Optional[int] = None


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    save_plots: bool = True
    show_plots: bool = False
    plot_format: str = "png"
    dpi: int = 300
    figure_size: tuple = (10, 8)


@dataclass
class MainConfig:
    """Main configuration class that matches training script expectations."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def validate(self) -> None:
        """Validate configuration."""
        # Validate model-specific requirements
        if self.model.name == "transformer" and self.model.num_heads <= 0:
            raise ValueError("Transformer model requires num_heads > 0")

        # Validate paths
        if not Path(self.data.annotation_path).exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.data.annotation_path}"
            )

        # Validate cross-validation
        if self.cross_validation.n_splits < 2:
            raise ValueError("Cross-validation requires at least 2 splits")

        # Auto-set random state if not provided
        if self.cross_validation.random_state is None:
            self.cross_validation.random_state = self.experiment.seed


# Alias for backward compatibility
Config = MainConfig


def load_config(config_path: str) -> MainConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Loaded configuration object.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create nested config objects
    experiment = ExperimentConfig(**config_dict.get("experiment", {}))
    data = DataConfig(**config_dict.get("data", {}))
    model = ModelConfig(**config_dict.get("model", {}))
    training = TrainingConfig(**config_dict.get("training", {}))
    cross_validation = CrossValidationConfig(**config_dict.get("cross_validation", {}))
    visualization = VisualizationConfig(**config_dict.get("visualization", {}))

    # Create main config
    config = MainConfig(
        experiment=experiment,
        data=data,
        model=model,
        training=training,
        cross_validation=cross_validation,
        visualization=visualization,
    )

    # Validate configuration
    config.validate()

    return config
