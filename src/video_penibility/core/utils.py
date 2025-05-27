"""Core utility functions."""

import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Union, Optional


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional file to write logs to.
        format_string: Custom format string for log messages.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),  # Console output
        ],
    )

    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device based on string specification.

    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.).

    Returns:
        torch.device object.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # Verify device is available
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device


def create_output_directory(base_path: Union[str, Path], experiment_name: str) -> Path:
    """Create output directory for experiment.

    Args:
        base_path: Base directory path.
        experiment_name: Name of the experiment.

    Returns:
        Path to created output directory.
    """
    import datetime

    base_path = Path(base_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_path / f"{experiment_name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def save_model_summary(model: torch.nn.Module, output_path: Union[str, Path]) -> None:
    """Save model summary to file.

    Args:
        model: PyTorch model.
        output_path: Path to save summary.
    """
    output_path = Path(output_path)

    total_params, trainable_params = count_parameters(model)

    summary = f"""Model Summary
{'=' * 50}
Model Class: {model.__class__.__name__}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Non-trainable Parameters: {total_params - trainable_params:,}

Architecture:
{str(model)}
"""

    with open(output_path, "w") as f:
        f.write(summary)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
