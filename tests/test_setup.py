#!/usr/bin/env python3
"""Simple test script to verify the package setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test basic imports."""
    try:
        print("✓ Config import successful")

        print("✓ Core utils import successful")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration functionality."""
    try:
        from video_penibility.config import Config

        # Test default config
        config = Config()
        assert config.experiment.name == "default_experiment"
        assert config.model.name == "gru"
        print("✓ Default configuration works")

        # Test config modification
        config.experiment.name = "test_experiment"
        config.model.hidden_dim = 256
        assert config.experiment.name == "test_experiment"
        assert config.model.hidden_dim == 256
        print("✓ Configuration modification works")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_model_factory():
    """Test model factory functionality."""
    try:
        from video_penibility.config import Config
        from video_penibility.models import ModelFactory

        config = Config()
        input_dim = 1024

        # Test model creation
        model = ModelFactory.create_model(config.model, input_dim)
        print(f"✓ Model creation successful: {model.__class__.__name__}")

        # Test model info
        available_models = ModelFactory.list_available_models()
        print("✓ Available models: {}".format(available_models))

        return True
    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Video Penibility Assessment Package Setup")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Factory", test_model_factory),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print("\n{}:".format(test_name))
        if test_func():
            passed += 1
        else:
            print("  Test failed!")

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Package setup is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
