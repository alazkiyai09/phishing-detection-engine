"""
Configuration management for multi-agent phishing detector.
"""
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default 'config.yaml'

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        config_path = "config.yaml"

    path = Path(config_path)

    if path.exists():
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}

    return {}


def get_agent_weights(config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Get agent weights from configuration.

    Args:
        config: Configuration dictionary. If None, loads from default path.

    Returns:
        Dictionary mapping agent names to their weights
    """
    if config is None:
        config = get_config()

    # Try to get weights from config
    weights = config.get("coordinator", {}).get("agent_weights")

    if weights is not None:
        return weights

    # Default weights
    return {
        "url_analyst": 1.0,
        "content_analyst": 1.2,
        "header_analyst": 1.1,
        "visual_analyst": 0.8,
        "financial_specialist": 1.3
    }
