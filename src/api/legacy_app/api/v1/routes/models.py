"""
Model information endpoints.
"""
from fastapi import APIRouter

from src.api.legacy_app.schemas.responses import ModelsListResponse, ModelInfo
from src.api.legacy_app.schemas.enums import ModelType
from src.api.legacy_app.config import settings
from src.api.legacy_app.services.model_service import get_xgboost_model, get_transformer_model, get_available_models

router = APIRouter()


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available models and their performance metrics.

    Returns information about each model including availability,
    performance metrics, and latency characteristics.
    """
    models_list = []

    # Get XGBoost model info
    xgboost_model = get_xgboost_model()
    if xgboost_model and xgboost_model.is_loaded:
        try:
            info = xgboost_model.get_info()
            models_list.append(ModelInfo(
                name="xgboost",
                type=ModelType.XGBOOST,
                available=True,
                description="XGBoost Gradient Boosting - Classical ML with 60+ features",
                version=info.get("version", "unknown"),
                performance=info.get("performance", {}),
                avg_latency_ms=info.get("avg_latency_ms", 15.0),  # Expected ~15ms
                last_trained=info.get("training_date", None)
            ))
        except Exception as e:
            # Model loaded but info failed
            models_list.append(ModelInfo(
                name="xgboost",
                type=ModelType.XGBOOST,
                available=True,
                description="XGBoost model (info unavailable)",
                version="unknown",
                performance={},
                avg_latency_ms=None
            ))

    # Get Transformer model info
    transformer_model = get_transformer_model()
    if transformer_model and transformer_model.is_loaded:
        try:
            info = transformer_model.get_info()
            models_list.append(ModelInfo(
                name="transformer",
                type=ModelType.TRANSFORMER,
                available=True,
                description=f"Transformer model - {info.get('architecture', 'DistilBERT')}",
                version=info.get("version", "unknown"),
                performance=info.get("performance", {}),
                avg_latency_ms=info.get("avg_latency_ms", 500.0),  # Expected ~500ms
                last_trained=info.get("training_date", None)
            ))
        except Exception as e:
            models_list.append(ModelInfo(
                name="transformer",
                type=ModelType.TRANSFORMER,
                available=True,
                description="Transformer model (info unavailable)",
                version="unknown",
                performance={},
                avg_latency_ms=None
            ))

    # Add placeholder for multi-agent
    if settings.MULTI_AGENT_AVAILABLE:
        models_list.append(ModelInfo(
            name="multi_agent",
            type=ModelType.MULTI_AGENT,
            available=True,
            description="Multi-agent system with GLM-powered analysis",
            version="1.0",
            performance={},
            avg_latency_ms=3000.0  # Expected ~3s
        ))

    return ModelsListResponse(
        models=models_list,
        operating_mode=settings._get_operating_mode(),
        ensemble_weights=settings.get_ensemble_weights()
    )
