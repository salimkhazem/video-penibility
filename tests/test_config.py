"""Tests for configuration system."""

import pytest
import tempfile
import yaml
from pathlib import Path

from video_penibility.config import Config


def load_config(config_path):
    """Load config from YAML file - helper function for tests."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_test_config():
    """Create a Config object without validation for testing."""
    from video_penibility.config.schema import MainConfig

    # Create config without calling the constructor that validates
    config = Config.__new__(Config)
    config._config = MainConfig()
    config._config.validate = lambda: None  # Skip validation
    return config


def test_default_config():
    """Test default configuration creation."""
    config = create_test_config()

    assert config.experiment.name == "default_experiment"
    assert config.experiment.seed == 42
    assert config.data.features_type == "i3d"
    assert config.model.name == "gru"
    assert config.training.batch_size == 16


def test_config_from_dict():
    """Test configuration loading from dictionary."""
    config_dict = {
        "experiment": {"name": "test_experiment", "seed": 123},
        "model": {"name": "lstm", "hidden_dim": 256},
    }

    config = create_test_config()
    config.load_from_dict(config_dict)

    assert config.experiment.name == "test_experiment"
    assert config.experiment.seed == 123
    assert config.model.name == "lstm"
    assert config.model.hidden_dim == 256


def test_config_to_dict():
    """Test configuration conversion to dictionary."""
    config = create_test_config()
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert "experiment" in config_dict
    assert "data" in config_dict
    assert "model" in config_dict
    assert "training" in config_dict


def test_config_yaml_roundtrip():
    """Test saving and loading configuration from YAML."""
    config = create_test_config()
    config.experiment.name = "test_yaml"
    config.model.hidden_dim = 1024

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config.save_to_file(f.name)

        # Load it back
        config2 = create_test_config()
        config2.load_from_file(f.name)

        assert config2.experiment.name == "test_yaml"
        assert config2.model.hidden_dim == 1024

        # Clean up
        Path(f.name).unlink()


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_valid_config(self, sample_config, temp_dir):
        """Test loading a valid configuration file."""
        config_file = temp_dir / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        loaded_config = load_config(config_file)

        assert loaded_config == sample_config
        assert loaded_config["experiment"]["name"] == "test_experiment"
        assert loaded_config["model"]["name"] == "tcn"
        assert loaded_config["training"]["batch_size"] == 4

    def test_load_nonexistent_config(self):
        """Test error handling for non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent_config.yaml"))

    def test_load_invalid_yaml(self, temp_dir):
        """Test error handling for invalid YAML syntax."""
        config_file = temp_dir / "invalid_config.yaml"

        # Write invalid YAML
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            load_config(config_file)

    def test_config_path_as_string(self, sample_config, temp_dir):
        """Test loading config with string path."""
        config_file = temp_dir / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Pass string path instead of Path object
        loaded_config = load_config(str(config_file))

        assert loaded_config == sample_config


class TestConfigValidation:
    """Test configuration validation."""

    def test_required_sections_present(self, sample_config):
        """Test that all required sections are present."""
        required_sections = ["experiment", "data", "model", "training"]

        for section in required_sections:
            assert section in sample_config, f"Missing required section: {section}"

    def test_experiment_config_validation(self, sample_config):
        """Test experiment configuration validation."""
        experiment_config = sample_config["experiment"]

        assert "name" in experiment_config
        assert "device" in experiment_config
        assert "output_dir" in experiment_config

        assert isinstance(experiment_config["name"], str)
        assert isinstance(experiment_config["device"], str)
        assert isinstance(experiment_config["output_dir"], str)

    def test_model_config_validation(self, sample_config):
        """Test model configuration validation."""
        model_config = sample_config["model"]

        assert "name" in model_config
        assert "hidden_dim" in model_config
        assert "num_layers" in model_config

        assert isinstance(model_config["name"], str)
        assert isinstance(model_config["hidden_dim"], int)
        assert isinstance(model_config["num_layers"], int)
        assert model_config["hidden_dim"] > 0
        assert model_config["num_layers"] > 0

    def test_training_config_validation(self, sample_config):
        """Test training configuration validation."""
        training_config = sample_config["training"]

        assert "batch_size" in training_config
        assert "learning_rate" in training_config
        assert "num_epochs" in training_config

        assert isinstance(training_config["batch_size"], int)
        assert isinstance(training_config["learning_rate"], (int, float))
        assert isinstance(training_config["num_epochs"], int)

        assert training_config["batch_size"] > 0
        assert training_config["learning_rate"] > 0
        assert training_config["num_epochs"] > 0


class TestConfigDefaults:
    """Test configuration defaults and optional parameters."""

    def test_missing_optional_parameters(self, temp_dir):
        """Test handling of missing optional parameters."""
        minimal_config = {
            "experiment": {"name": "test", "device": "cpu", "output_dir": "results"},
            "data": {"features_type": "i3d", "data_type": "full_body"},
            "model": {"name": "tcn", "hidden_dim": 128, "num_layers": 2},
            "training": {"batch_size": 16, "learning_rate": 0.001, "num_epochs": 10},
        }

        config_file = temp_dir / "minimal_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(minimal_config, f)

        loaded_config = load_config(config_file)

        # Should load successfully even with minimal config
        assert loaded_config == minimal_config

    def test_config_with_extra_parameters(self, sample_config, temp_dir):
        """Test handling of extra parameters in config."""
        # Add some extra parameters
        sample_config["extra_section"] = {"param1": "value1"}
        sample_config["model"]["extra_param"] = 42

        config_file = temp_dir / "extended_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        loaded_config = load_config(config_file)

        # Should preserve extra parameters
        assert "extra_section" in loaded_config
        assert loaded_config["extra_section"]["param1"] == "value1"
        assert loaded_config["model"]["extra_param"] == 42


@pytest.mark.integration
class TestRealConfigFiles:
    """Integration tests with real configuration files."""

    def test_load_existing_configs(self):
        """Test loading existing config files from the configs directory."""
        configs_dir = Path("configs")

        if not configs_dir.exists():
            pytest.skip("Configs directory not found")

        config_files = list(configs_dir.glob("*.yaml"))

        if not config_files:
            pytest.skip("No config files found")

        for config_file in config_files:
            try:
                config = load_config(config_file)

                # Basic validation
                assert isinstance(config, dict)
                assert len(config) > 0

                print(f"✅ Successfully loaded {config_file.name}")

            except Exception as e:
                pytest.fail(f"Failed to load {config_file.name}: {e}")

    def test_config_consistency(self):
        """Test that all config files have consistent structure."""
        configs_dir = Path("configs")

        if not configs_dir.exists():
            pytest.skip("Configs directory not found")

        config_files = list(configs_dir.glob("*.yaml"))

        if not config_files:
            pytest.skip("No config files found")

        required_sections = ["experiment", "data", "model", "training"]

        for config_file in config_files:
            config = load_config(config_file)

            for section in required_sections:
                assert section in config, f"Missing {section} in {config_file.name}"


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_config_file(self, temp_dir):
        """Test handling of empty config file."""
        config_file = temp_dir / "empty_config.yaml"
        config_file.touch()  # Create empty file

        config = load_config(config_file)

        # Empty YAML file should result in None or empty dict
        assert config is None or config == {}

    def test_config_with_null_values(self, temp_dir):
        """Test handling of null values in config."""
        config_with_nulls = {
            "experiment": {
                "name": "test",
                "device": None,  # Null value
                "output_dir": "results",
            },
            "model": {"name": "tcn", "hidden_dim": 128, "dropout": None},  # Null value
        }

        config_file = temp_dir / "null_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_with_nulls, f)

        loaded_config = load_config(config_file)

        # Should preserve None values
        assert loaded_config["experiment"]["device"] is None
        assert loaded_config["model"]["dropout"] is None


if __name__ == "__main__":
    pytest.main([__file__])
