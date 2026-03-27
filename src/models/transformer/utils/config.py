"""
Configuration management for training experiments.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training transformer models."""

    # Model configuration
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    dropout: float = 0.1

    # LoRA configuration (for LoRA models only)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "auprc"

    # Data configuration
    max_length: int = 512
    truncation_strategy: str = "head_tail"  # 'head_only', 'tail_only', 'head_tail'
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2

    # Special tokens
    use_special_tokens: bool = True
    special_tokens: List[str] = field(default_factory=lambda: ["[SUBJECT]", "[BODY]", "[URL]", "[SENDER]"])

    # Mixed precision
    fp16: bool = True

    # Reproducibility
    seed: int = 42

    # Logging and checkpoints
    logging_steps: int = 50
    save_steps: int = 500
    output_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "phishing-detection-transformers"
    wandb_entity: Optional[str] = None

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'

    # Data paths
    data_path: str = "data/raw/phishing_emails_hf.csv"
    cache_dir: str = "data/cache"

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        assert self.truncation_strategy in ['head_only', 'tail_only', 'head_tail'], \
            "Invalid truncation strategy"
        assert self.early_stopping_metric in ['auprc', 'auroc', 'loss', 'accuracy'], \
            "Invalid early stopping metric"

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            TrainingConfig instance
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        print(f"💾 Configuration saved to {path}")

    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class ModelConfig:
    """Configuration for individual models."""

    name: str
    checkpoint: str
    use_lora: bool = False

    # Model-specific overrides
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None

    def to_training_config(self, base: TrainingConfig) -> TrainingConfig:
        """
        Convert to TrainingConfig with overrides applied.

        Args:
            base: Base TrainingConfig

        Returns:
            New TrainingConfig with this model's overrides
        """
        config_dict = base.__dict__.copy()
        config_dict['model_name'] = self.checkpoint
        config_dict['use_lora'] = self.use_lora

        # Apply overrides
        if self.learning_rate is not None:
            config_dict['learning_rate'] = self.learning_rate
        if self.batch_size is not None:
            config_dict['batch_size'] = self.batch_size
        if self.num_epochs is not None:
            config_dict['num_epochs'] = self.num_epochs

        return TrainingConfig(**config_dict)


# Predefined model configurations
MODEL_CONFIGS = {
    'bert': ModelConfig(
        name='BERT',
        checkpoint='bert-base-uncased',
        use_lora=False
    ),
    'roberta': ModelConfig(
        name='RoBERTa',
        checkpoint='roberta-base',
        use_lora=False
    ),
    'distilbert': ModelConfig(
        name='DistilBERT',
        checkpoint='distilbert-base-uncased',
        use_lora=False
    ),
    'lora-bert': ModelConfig(
        name='LoRA-BERT',
        checkpoint='bert-base-uncased',
        use_lora=True
    ),
}
