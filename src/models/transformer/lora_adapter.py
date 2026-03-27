from dataclasses import dataclass


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
