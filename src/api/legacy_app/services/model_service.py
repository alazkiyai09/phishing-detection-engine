"""
Model service for managing loaded ML models.
"""
import logging
from typing import Optional, Dict, Any

from src.api.legacy_app.models.xgboost_model import XGBoostModel
from src.api.legacy_app.models.transformer_model import TransformerModel
from src.api.legacy_app.models.multi_agent_model import MultiAgentModel
from src.api.legacy_app.config import settings
from src.api.legacy_app.utils.logger import get_logger

logger = get_logger(__name__)

# Global model instances
_xgboost_model: Optional[XGBoostModel] = None
_transformer_model: Optional[TransformerModel] = None
_multi_agent_model: Optional[MultiAgentModel] = None


def initialize_models():
    """Initialize all available models."""
    global _xgboost_model, _transformer_model, _multi_agent_model

    # Load XGBoost
    try:
        xgboost_path = f"{settings.MODELS_BASE_PATH}/day2_xgboost/xgboost_phishing_classifier.json"
        _xgboost_model = XGBoostModel(xgboost_path)
        if _xgboost_model.is_loaded:
            settings.XGBOOST_AVAILABLE = True
            logger.info("✅ XGBoost model loaded and available")
        else:
            logger.warning("XGBoost model loaded but not available")
            _xgboost_model = None
    except Exception as e:
        logger.warning(f"Failed to load XGBoost model: {e}")
        _xgboost_model = None

    # Load Transformer
    try:
        transformer_path = f"{settings.MODELS_BASE_PATH}/day3_distilbert"
        from pathlib import Path
        if (Path(transformer_path) / "config.json").exists():
            _transformer_model = TransformerModel(transformer_path)
            if _transformer_model.is_loaded:
                settings.TRANSFORMER_AVAILABLE = True
                logger.info("✅ Transformer model loaded and available")
            else:
                logger.warning("Transformer model loaded but not available")
                _transformer_model = None
        else:
            logger.info("Transformer model not found, skipping")
    except Exception as e:
        logger.warning(f"Failed to load Transformer model: {e}")
        _transformer_model = None

    # Load Multi-Agent (if GLM API key is available)
    try:
        _multi_agent_model = MultiAgentModel()
        if _multi_agent_model.is_available:
            settings.MULTI_AGENT_AVAILABLE = True
            logger.info("✅ Multi-agent model loaded and available")
        else:
            logger.info("Multi-agent model not available (check GLM_API_KEY)")
            _multi_agent_model = None
    except Exception as e:
        logger.warning(f"Failed to load Multi-agent model: {e}")
        _multi_agent_model = None

    logger.info(f"Model initialization complete. XGBoost: {settings.XGBOOST_AVAILABLE}, Transformer: {settings.TRANSFORMER_AVAILABLE}, Multi-Agent: {settings.MULTI_AGENT_AVAILABLE}")


def get_xgboost_model() -> Optional[XGBoostModel]:
    """Get the XGBoost model instance."""
    return _xgboost_model


def get_transformer_model() -> Optional[TransformerModel]:
    """Get the Transformer model instance."""
    return _transformer_model


def predict_with_xgboost(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Make prediction using XGBoost model.

    Args:
        features: Feature dictionary

    Returns:
        Prediction result or None if model unavailable
    """
    model = get_xgboost_model()
    if model and model.is_loaded:
        try:
            return model.predict(features)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}", exc_info=True)
    return None


def predict_with_transformer(email_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Make prediction using Transformer model.

    Args:
        email_data: Email data dictionary

    Returns:
        Prediction result or None if model unavailable
    """
    model = get_transformer_model()
    if model and model.is_loaded:
        try:
            return model.predict(email_data)
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}", exc_info=True)
    return None


def get_available_models() -> Dict[str, bool]:
    """
    Get availability status of all models.

    Returns:
        Dictionary with model availability
    """
    return {
        "xgboost": _xgboost_model is not None and _xgboost_model.is_loaded,
        "transformer": _transformer_model is not None and _transformer_model.is_loaded,
        "multi_agent": settings.MULTI_AGENT_AVAILABLE
    }


def get_model_info(model_type: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a model.

    Args:
        model_type: Type of model (xgboost, transformer, or multi_agent)

    Returns:
        Model information dictionary or None
    """
    if model_type == "xgboost":
        model = get_xgboost_model()
        if model:
            return model.get_info()
    elif model_type == "transformer":
        model = get_transformer_model()
        if model:
            return model.get_info()
    elif model_type == "multi_agent":
        model = get_multi_agent_model()
        if model:
            return model.get_info()
    return None


def get_multi_agent_model() -> Optional[MultiAgentModel]:
    """Get the Multi-Agent model instance."""
    return _multi_agent_model


def predict_with_multi_agent(email_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Make prediction using Multi-Agent model.

    Args:
        email_data: Email data dictionary

    Returns:
        Prediction result or None if model unavailable
    """
    model = get_multi_agent_model()
    if model and model.is_loaded:
        try:
            return model.predict(email_data)
        except Exception as e:
            logger.error(f"Multi-agent prediction failed: {e}", exc_info=True)
    return None
