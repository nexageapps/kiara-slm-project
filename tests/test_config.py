"""Tests for configuration management."""

import pytest
import tempfile
from pathlib import Path
from kiara.config import Config, ModelConfig, TrainingConfig


def test_config_creation():
    """Test config can be created with defaults."""
    config = Config()
    
    assert config.model is not None
    assert config.training is not None
    assert config.data is not None
    assert config.logging is not None


def test_config_to_dict():
    """Test config can be converted to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "model" in config_dict
    assert "training" in config_dict


def test_config_save_load_yaml():
    """Test config can be saved and loaded from YAML."""
    config = Config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save(f.name)
        temp_path = f.name
    
    try:
        loaded_config = Config.from_yaml(temp_path)
        assert loaded_config.model.vocab_size == config.model.vocab_size
        assert loaded_config.training.learning_rate == config.training.learning_rate
    finally:
        Path(temp_path).unlink()


def test_config_save_load_json():
    """Test config can be saved and loaded from JSON."""
    config = Config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config.save(f.name)
        temp_path = f.name
    
    try:
        loaded_config = Config.from_json(temp_path)
        assert loaded_config.model.vocab_size == config.model.vocab_size
        assert loaded_config.training.learning_rate == config.training.learning_rate
    finally:
        Path(temp_path).unlink()
