"""Configuration management for Kiara SLM."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import json


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 5e-4
    batch_size: int = 8
    num_epochs: int = 10
    weight_decay: float = 0.1
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    eval_freq: int = 100
    eval_iter: int = 10
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 3


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    train_data_path: Optional[Path] = None
    val_data_path: Optional[Path] = None
    stride: int = 128
    num_workers: int = 4


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    use_wandb: bool = False
    wandb_project: str = "kiara-slm"
    wandb_entity: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config,
            checkpoint_dir=Path(config_dict.get("checkpoint_dir", "./checkpoints")),
            device=config_dict.get("device", "cuda"),
            mixed_precision=config_dict.get("mixed_precision", True),
            seed=config_dict.get("seed", 42),
        )
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": {k: str(v) if isinstance(v, Path) else v 
                    for k, v in self.data.__dict__.items()},
            "logging": {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.logging.__dict__.items()},
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
        }
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")


def load_config_from_env() -> Config:
    """Load configuration from environment variables."""
    config = Config()
    
    # Model config
    if os.getenv("MODEL_SIZE"):
        size = os.getenv("MODEL_SIZE")
        if size == "small":
            config.model = ModelConfig(emb_dim=768, n_heads=12, n_layers=12)
        elif size == "medium":
            config.model = ModelConfig(emb_dim=1024, n_heads=16, n_layers=24)
        elif size == "large":
            config.model = ModelConfig(emb_dim=1280, n_heads=20, n_layers=36)
    
    if os.getenv("CONTEXT_LENGTH"):
        config.model.context_length = int(os.getenv("CONTEXT_LENGTH"))
    
    # Training config
    if os.getenv("LEARNING_RATE"):
        config.training.learning_rate = float(os.getenv("LEARNING_RATE"))
    if os.getenv("BATCH_SIZE"):
        config.training.batch_size = int(os.getenv("BATCH_SIZE"))
    if os.getenv("NUM_EPOCHS"):
        config.training.num_epochs = int(os.getenv("NUM_EPOCHS"))
    
    # Data config
    if os.getenv("DATA_DIR"):
        config.data.data_dir = Path(os.getenv("DATA_DIR"))
    if os.getenv("TRAIN_DATA_PATH"):
        config.data.train_data_path = Path(os.getenv("TRAIN_DATA_PATH"))
    if os.getenv("VAL_DATA_PATH"):
        config.data.val_data_path = Path(os.getenv("VAL_DATA_PATH"))
    
    # Logging config
    if os.getenv("LOG_LEVEL"):
        config.logging.log_level = os.getenv("LOG_LEVEL")
    if os.getenv("USE_WANDB"):
        config.logging.use_wandb = os.getenv("USE_WANDB").lower() == "true"
    if os.getenv("WANDB_PROJECT"):
        config.logging.wandb_project = os.getenv("WANDB_PROJECT")
    
    # Device config
    if os.getenv("DEVICE"):
        config.device = os.getenv("DEVICE")
    if os.getenv("MIXED_PRECISION"):
        config.mixed_precision = os.getenv("MIXED_PRECISION").lower() == "true"
    
    return config
