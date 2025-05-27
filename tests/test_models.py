"""Tests for model factory and core models."""

import pytest
import torch

from video_penibility.models.factory import ModelFactory
from video_penibility.models.transformer import TransformerModel
from video_penibility.models.tcn import TCNModel
from video_penibility.models.rnn import LSTMModel, GRUModel
from video_penibility.config.schema import ModelConfig


class TestModelFactory:
    """Test the model factory."""

    def test_create_transformer_model(self):
        """Test creating a transformer model."""
        factory = ModelFactory()
        model_config = ModelConfig(
            name="transformer",
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
        )

        model = factory.create_model(model_config, input_dim=1024)

        assert isinstance(model, TransformerModel)
        assert model.input_dim == 1024
        assert model.hidden_dim == 256
        assert model.num_layers == 2

    def test_create_tcn_model(self):
        """Test creating a TCN model."""
        factory = ModelFactory()
        model_config = ModelConfig(
            name="tcn",
            hidden_dim=128,
            kernel_size=3,
            dropout=0.2,
        )

        model = factory.create_model(model_config, input_dim=1024)

        assert isinstance(model, TCNModel)
        assert model.input_dim == 1024
        assert model.hidden_dim == 128

    def test_create_lstm_model(self):
        """Test creating an LSTM model."""
        factory = ModelFactory()
        model_config = ModelConfig(
            name="lstm",
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
        )

        model = factory.create_model(model_config, input_dim=768)

        assert isinstance(model, LSTMModel)
        assert model.input_dim == 768
        assert model.hidden_dim == 256
        assert model.num_layers == 2

    def test_create_gru_model(self):
        """Test creating a GRU model."""
        factory = ModelFactory()
        model_config = ModelConfig(
            name="gru",
            hidden_dim=128,
            num_layers=1,
            dropout=0.0,
        )

        model = factory.create_model(model_config, input_dim=512)

        assert isinstance(model, GRUModel)
        assert model.input_dim == 512
        assert model.hidden_dim == 128
        assert model.num_layers == 1

    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        factory = ModelFactory()
        model_config = ModelConfig(name="invalid_model")

        with pytest.raises(ValueError, match="Unsupported model type"):
            factory.create_model(model_config, input_dim=1024)


class TestTransformerModel:
    """Test the Transformer model."""

    def test_forward_pass(self, sample_features):
        """Test forward pass of transformer model."""
        model = TransformerModel(
            input_dim=1024, hidden_dim=256, num_layers=2, num_heads=8, dropout=0.1
        )

        outputs = model(sample_features)

        assert outputs.shape == (sample_features.size(0), 1)  # (batch_size, output_dim)
        assert torch.all(torch.isfinite(outputs))  # Should produce finite values

    def test_model_parameters(self):
        """Test model parameter initialization."""
        model = TransformerModel(
            input_dim=1024, hidden_dim=128, num_layers=1, num_heads=4
        )

        # Check that model has parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

        # Check that most parameters are properly initialized (not all zeros)
        non_zero_params = 0
        total_param_count = 0
        for param in model.parameters():
            if param.numel() > 1:  # Skip bias terms which might be zero
                non_zero_params += torch.count_nonzero(param).item()
                total_param_count += param.numel()

        # At least 50% of parameters should be non-zero
        assert non_zero_params / total_param_count > 0.5


class TestTCNModel:
    """Test the TCN model."""

    def test_forward_pass(self, sample_features):
        """Test forward pass of TCN model."""
        model = TCNModel(input_dim=1024, hidden_dim=128, kernel_size=3, dropout=0.1)

        outputs = model(sample_features)

        assert outputs.shape == (sample_features.size(0), 1)  # (batch_size, output_dim)
        assert torch.all(torch.isfinite(outputs))  # Should produce finite values

    def test_different_kernel_sizes(self, sample_features):
        """Test TCN with different kernel sizes."""
        for kernel_size in [3, 5, 7]:
            model = TCNModel(input_dim=1024, hidden_dim=64, kernel_size=kernel_size)

            outputs = model(sample_features)
            assert outputs.shape == (sample_features.size(0), 1)


class TestLSTMModel:
    """Test the LSTM model."""

    def test_forward_pass(self, sample_features):
        """Test forward pass of LSTM model."""
        model = LSTMModel(input_dim=1024, hidden_dim=256, num_layers=2, dropout=0.1)

        outputs = model(sample_features)

        assert outputs.shape == (sample_features.size(0), 1)  # (batch_size, output_dim)
        assert torch.all(torch.isfinite(outputs))  # Should produce finite values

    def test_bidirectional_lstm(self, sample_features):
        """Test bidirectional LSTM."""
        model = LSTMModel(
            input_dim=1024, hidden_dim=128, num_layers=1, bidirectional=True
        )

        outputs = model(sample_features)
        assert outputs.shape == (sample_features.size(0), 1)


class TestGRUModel:
    """Test the GRU model."""

    def test_forward_pass(self, sample_features):
        """Test forward pass of GRU model."""
        model = GRUModel(input_dim=1024, hidden_dim=128, num_layers=1, dropout=0.0)

        outputs = model(sample_features)

        assert outputs.shape == (sample_features.size(0), 1)  # (batch_size, output_dim)
        assert torch.all(torch.isfinite(outputs))  # Should produce finite values


class TestModelIntegration:
    """Integration tests for models."""

    @pytest.mark.parametrize("model_name", ["transformer", "tcn", "lstm", "gru"])
    def test_model_training_mode(self, model_name, sample_features):
        """Test that all models can switch between train/eval modes."""
        factory = ModelFactory()
        model_config = ModelConfig(
            name=model_name,
            hidden_dim=64,
            num_layers=1,
        )
        if model_name == "transformer":
            model_config.num_heads = 4
        elif model_name == "tcn":
            model_config.kernel_size = 3

        model = factory.create_model(model_config, input_dim=1024)

        # Test training mode
        model.train()
        assert model.training

        outputs_train = model(sample_features)

        # Test evaluation mode
        model.eval()
        assert not model.training

        outputs_eval = model(sample_features)

        # Outputs should have the same shape regardless of mode
        assert outputs_train.shape == outputs_eval.shape

    def test_model_gradient_computation(self, sample_features, sample_targets):
        """Test that models can compute gradients."""
        model = TransformerModel(
            input_dim=1024, hidden_dim=64, num_layers=1, num_heads=4
        )

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # Forward pass
        outputs = model(sample_features)
        loss = criterion(outputs, sample_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check that gradients are computed
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        assert has_gradients, "Model should have non-zero gradients after backward pass"


@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for models."""

    def test_inference_speed(self, sample_features, performance_threshold):
        """Test model inference speed."""
        import time

        model = TransformerModel(
            input_dim=1024, hidden_dim=128, num_layers=2, num_heads=8
        )
        model.eval()

        # Warm up
        with torch.no_grad():
            _ = model(sample_features)

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            _ = model(sample_features)
        end_time = time.time()

        inference_time_ms = (end_time - start_time) * 1000
        assert inference_time_ms < performance_threshold["max_inference_time_ms"]

    def test_memory_usage(self, sample_features, performance_threshold):
        """Test model memory usage."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            model = TransformerModel(
                input_dim=1024, hidden_dim=512, num_layers=4, num_heads=8
            )

            # Forward pass
            with torch.no_grad():
                _ = model(sample_features)

            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_memory - initial_memory

            assert memory_usage < performance_threshold["max_memory_mb"]

        except ImportError:
            pytest.skip("psutil not available for memory testing")
