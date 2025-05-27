"""
Video Penibility Assessment Package

A modular framework for assessing physical workload (penibility) from video sequences
using deep learning techniques.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.experiment import ExperimentRunner
from .config import Config
from .models import ModelFactory
from .datasets import DatasetFactory
from .trainers import TrainerFactory

__all__ = [
    "ExperimentRunner",
    "Config",
    "ModelFactory",
    "DatasetFactory",
    "TrainerFactory",
]
