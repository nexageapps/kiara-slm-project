"""Kiara - Small Language Model Package."""

__version__ = "0.1.0"

from kiara.model import GPTModel, get_gpt_config
from kiara.training import train_model_simple, generate_text_simple

__all__ = [
    "GPTModel",
    "get_gpt_config",
    "train_model_simple",
    "generate_text_simple",
]
