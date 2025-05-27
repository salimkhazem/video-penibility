"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any, Generator
import numpy as np


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "experiment": {
            "name": "test_experiment",
            "device": "cpu",
            "output_dir": "test_results",
        },
        "data": {
            "features_type": "i3d",
            "data_type": "full_body",
            "annotation_path": "/path/to/test/annotations.csv",
            "features_root": "/path/to/test/features/",
        },
        "model": {"name": "tcn", "hidden_dim": 128, "num_layers": 2, "dropout": 0.1},
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "num_epochs": 5,
            "optimizer": "adam",
            "n_splits": 2,
        },
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_features() -> torch.Tensor:
    """Sample feature tensor for testing."""
    # Simulate I3D features: (batch_size, seq_len, feature_dim)
    batch_size, seq_len, feature_dim = 2, 10, 1024
    return torch.randn(batch_size, seq_len, feature_dim)


@pytest.fixture
def sample_targets() -> torch.Tensor:
    """Sample target values for testing."""
    # Penibility scores between 1-10
    batch_size = 2
    return torch.rand(batch_size) * 9 + 1  # Values between 1-10


@pytest.fixture
def sample_sequence_lengths() -> torch.Tensor:
    """Sample sequence lengths for testing."""
    return torch.tensor([8, 10])  # Variable sequence lengths


@pytest.fixture
def device() -> torch.device:
    """Device for testing (CPU only in CI)."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def mock_dataset_file(temp_dir: Path) -> Path:
    """Create a mock CSV dataset file for testing."""
    csv_content = """video_id,subject_id,penibility_score,sequence_length
test_video_1,subject_1,5.5,25
test_video_2,subject_2,3.2,30
test_video_3,subject_1,7.8,28
test_video_4,subject_3,2.1,22
"""
    csv_file = temp_dir / "test_annotations.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def mock_features_dir(temp_dir: Path) -> Path:
    """Create mock feature files for testing."""
    features_dir = temp_dir / "features"
    features_dir.mkdir()

    # Create mock feature files
    for video_id in ["test_video_1", "test_video_2", "test_video_3", "test_video_4"]:
        feature_file = features_dir / f"{video_id}.npy"
        # Random features with variable length
        seq_len = np.random.randint(20, 35)
        features = np.random.randn(seq_len, 1024).astype(np.float32)
        np.save(feature_file, features)

    return features_dir


# Performance testing fixtures
@pytest.fixture
def performance_threshold():
    """Performance thresholds for testing."""
    return {
        "max_memory_mb": 1000,  # 1GB max memory usage
        "max_inference_time_ms": 100,  # 100ms max inference time
        "min_accuracy": 0.1,  # Minimum accuracy threshold
    }


# Mocking fixtures for external dependencies
@pytest.fixture
def mock_tensorboard_logger():
    """Mock TensorBoard logger for testing."""

    class MockLogger:
        def __init__(self):
            self.logged_scalars = {}
            self.logged_figures = {}

        def add_scalar(self, tag, value, step):
            if tag not in self.logged_scalars:
                self.logged_scalars[tag] = []
            self.logged_scalars[tag].append((value, step))

        def add_figure(self, tag, figure, step):
            if tag not in self.logged_figures:
                self.logged_figures[tag] = []
            self.logged_figures[tag].append((figure, step))

        def close(self):
            pass

    return MockLogger()


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "performance: marks performance tests")
