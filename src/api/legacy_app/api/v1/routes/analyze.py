"""
Email and URL analysis endpoints.
"""
import uuid
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import asyncio

from fastapi import APIRouter, HTTPException, Depends
from pydantic import ValidationError

from src.api.legacy_app.schemas.requests import EmailAnalysisRequest, URLAnalysisRequest, BatchAnalysisRequest
from src.api.legacy_app.schemas.responses import AnalysisResponse, BatchAnalysisResponse
from src.api.legacy_app.schemas.enums import Verdict, RiskLevel, ModelType
from src.api.legacy_app.services.url_analyzer import url_analyzer
from src.api.legacy_app.services.feature_extractor import feature_extraction_service
from src.api.legacy_app.services.cache import cache_service
from src.api.legacy_app.services.model_service import get_xgboost_model, get_transformer_model, predict_with_xgboost, predict_with_transformer, predict_with_multi_agent
from src.api.legacy_app.middleware.metrics import record_model_prediction, record_cache_hit, record_cache_miss
from src.api.legacy_app.config import settings
from src.api.legacy_app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _generate_email_id() -> str:
    """Generate unique email ID."""
    return f"email_{uuid.uuid4().hex[:12]}"


def _risk_score_to_level(risk_score: int) -> RiskLevel:
    """Convert risk score to risk level."""
    if risk_score >= 80:
        return RiskLevel.CRITICAL
    elif risk_score >= 60:
        return RiskLevel.HIGH
    elif risk_score >= 40:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


async def _extract_email_features(
    request: EmailAnalysisRequest
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract features from email based on request type.

    Args:
        request: Email analysis request

    Returns:
        Tuple of (parsed_email, features_dict)

    Raises:
        HTTPException: If feature extraction fails
    """
    if request.raw_email:
        extraction_result = await feature_extraction_service.extract_from_raw_email(
            request.raw_email
        )
        parsed_email = extraction_result["parsed_email"]
        features = extraction_result["features"]["features"]
    elif request.parsed_email:
        parsed_email = request.parsed_email
        extraction_result = await feature_extraction_service.extract_from_parsed_email(
            parsed_email
        )
        features = extraction_result["features"]["features"]
    else:
        raise HTTPException(status_code=400, detail="No email data provided")

    return parsed_email, features


async def _check_prediction_cache(
    request: EmailAnalysisRequest,
    features: Dict[str, Any]
) -> Optional[AnalysisResponse]:
    """
    Check if prediction result is cached.

    Args:
        request: Email analysis request
        features: Extracted features dict

    Returns:
        Cached response if found, None otherwise
    """
    if not request.use_cache or request.model_type == ModelType.ENSEMBLE:
        return None

    features_str = json.dumps(features, sort_keys=True)
    cache_key = f"prediction:{request.model_type.value}:{hashlib.md5(features_str.encode()).hexdigest()}"
    cached_result = await cache_service.get(cache_key)

    if cached_result:
        logger.info(f"Prediction cache hit for {request.model_type.value}")
        record_cache_hit("model_prediction")
        return AnalysisResponse(**cached_result)

    record_cache_miss("model_prediction")
    return None


async def _run_xgboost_prediction(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Run XGBoost model prediction."""
    if not settings.XGBOOST_AVAILABLE:
        raise HTTPException(status_code=503, detail="XGBoost model not available")
    return predict_with_xgboost(features)


async def _run_transformer_prediction(
    parsed_email: Dict[str, Any]
) -> Dict[str, Any]:
    """Run Transformer model prediction."""
    if not settings.TRANSFORMER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Transformer model not available")
    return predict_with_transformer(parsed_email)


async def _run_multi_agent_prediction(
    parsed_email: Dict[str, Any]
) -> Dict[str, Any]:
    """Run Multi-Agent model prediction."""
    if not settings.MULTI_AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-agent model not available")
    return predict_with_multi_agent(parsed_email)


def _calculate_ensemble_prediction(
    all_predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate weighted ensemble prediction from individual model predictions.

    Args:
        all_predictions: List of individual prediction dicts

    Returns:
        Ensemble prediction dict with verdict, confidence, risk_score
    """
    phishing_score = 0.0
    total_weight = 0.0
    weights = settings.get_ensemble_weights()

    for pred in all_predictions:
        model_name = pred.get("model_name", "unknown")
        weight = weights.get(model_name, 1.0 / len(all_predictions))

        if pred["verdict"] == "PHISHING":
            phishing_score += pred["confidence"] * weight
        else:
            phishing_score += (1 - pred["confidence"]) * weight

        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        phishing_score /= total_weight

    # Determine ensemble verdict
    if phishing_score >= 0.6:
        verdict = "PHISHING"
        confidence = phishing_score
        risk_score = int(phishing_score * 100)
    elif phishing_score >= 0.4:
        verdict = "SUSPICIOUS"
        confidence = phishing_score + 0.1
        risk_score = int(phishing_score * 80)
    else:
        verdict = "LEGITIMATE"
        confidence = 1 - phishing_score
        risk_score = int(phishing_score * 30)

    return {
        "model_name": "ensemble",
        "verdict": verdict,
        "confidence": confidence,
        "risk_score": risk_score,
        "individual_predictions": all_predictions
    }


async def _get_model_prediction(
    model_type: ModelType,
    features: Dict[str, Any],
    parsed_email: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run prediction based on model type.

    Args:
        model_type: Type of model to use
        features: Extracted features
        parsed_email: Parsed email data

    Returns:
        Tuple of (prediction_dict, list_of_individual_predictions)
    """
    individual_predictions = []

    if model_type == ModelType.XGBOOST:
        prediction = await _run_xgboost_prediction(features)
        individual_predictions.append(prediction)

    elif model_type == ModelType.TRANSFORMER:
        prediction = await _run_transformer_prediction(parsed_email)
        individual_predictions.append(prediction)

    elif model_type == ModelType.MULTI_AGENT:
        prediction = await _run_multi_agent_prediction(parsed_email)
        individual_predictions.append(prediction)

    elif model_type == ModelType.ENSEMBLE:
        all_predictions = []

        if settings.XGBOOST_AVAILABLE:
            all_predictions.append(await _run_xgboost_prediction(features))

        if settings.TRANSFORMER_AVAILABLE:
            all_predictions.append(await _run_transformer_prediction(parsed_email))

        if settings.MULTI_AGENT_AVAILABLE:
            all_predictions.append(await _run_multi_agent_prediction(parsed_email))

        if len(all_predictions) == 0:
            raise HTTPException(status_code=503, detail="No models available for ensemble")

        individual_predictions = all_predictions
        prediction = _calculate_ensemble_prediction(all_predictions)

    return prediction, individual_predictions


def _build_analysis_breakdown(
    features: Dict[str, Any],
    individual_predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build analysis breakdown from features and predictions."""
    return {
        "feature_count": len(features),
        "url_features": {k: v for k, v in features.items() if k.startswith("url_")},
        "content_features": {
            k: v for k, v in features.items()
            if "urgency" in k or "threat" in k or "financial" in k
        },
        "financial_indicators": {
            k: v for k, v in features.items()
            if "bank" in k or "wire" in k or "credential" in k
        },
        "individual_predictions": individual_predictions if len(individual_predictions) > 1 else None
    }


def _generate_explanation(
    verdict: str,
    model_name: str
) -> str:
    """Generate explanation based on verdict and model."""
    if verdict == "PHISHING":
        return f"High confidence phishing detection by {model_name} model. Multiple risk factors detected."
    elif verdict == "SUSPICIOUS":
        return f"Some phishing indicators detected by {model_name} model. Manual review recommended."
    else:
        return f"Email appears legitimate based on {model_name} analysis."


@router.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """
    Quick URL-only analysis.

    Faster than full email analysis, focuses on URL-based detection.
    Uses heuristic analysis without ML models for sub-100ms response times.
    """
    email_id = _generate_email_id()
    start_time = datetime.now()

    try:
        # Check cache first
        cache_key = None
        if request.use_cache:
            cache_key = cache_service.generate_url_key(request.url)
            cached_result = await cache_service.get(cache_key)

            if cached_result:
                logger.info(f"URL cache hit: {request.url}")
                record_cache_hit("url_reputation")
                return AnalysisResponse(**cached_result)

            record_cache_miss("url_reputation")

        # Analyze URL
        result = await url_analyzer.analyze_url(
            url=request.url,
            context=request.context
        )

        # Map to response format
        response = AnalysisResponse(
            email_id=email_id,
            verdict=Verdict(result["verdict"]),
            confidence=result["risk_score"] / 100.0,
            risk_score=result["risk_score"],
            risk_level=_risk_score_to_level(result["risk_score"]),
            model_used="url_heuristic",
            analysis={
                "url_risk": {
                    "url": request.url,
                    "checks": result["checks"],
                    "risk_factors": [r for r in result["explanation"].split(". ") if r]
                }
            },
            explanation=result["explanation"],
            processing_time_ms=result["processing_time_ms"],
            cache_hit=False,
            timestamp=datetime.utcnow().isoformat()
        )

        # Cache result
        if request.use_cache and cache_key:
            await cache_service.set(
                cache_key,
                response.model_dump(),
                ttl=settings.REDIS_URL_REPUTATION_TTL
            )

        # Record metrics
        record_model_prediction(
            model_type="url_heuristic",
            verdict=result["verdict"],
            duration_ms=result["processing_time_ms"]
        )

        return response

    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"URL analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="URL analysis failed")


@router.post("/analyze/email", response_model=AnalysisResponse)
async def analyze_email(request: EmailAnalysisRequest):
    """
    Analyze a single email for phishing indicators.

    Supports both raw EML format and pre-parsed email data.
    Uses XGBoost, Transformer, or Ensemble model based on model_type parameter.
    """
    email_id = _generate_email_id()
    start_time = datetime.now()

    # Check if feature extraction is available
    if not feature_extraction_service.is_available:
        raise HTTPException(
            status_code=503,
            detail="Feature extraction service not available. Ensure Day 1 pipeline is installed."
        )

    # Extract features from email
    parsed_email, features = await _extract_email_features(request)

    # Check cache
    cached_response = await _check_prediction_cache(request, features)
    if cached_response:
        return cached_response

    # Run prediction
    prediction, individual_predictions = await _get_model_prediction(
        request.model_type, features, parsed_email
    )

    # Extract results
    verdict_str = prediction.get("verdict", "SUSPICIOUS")
    confidence = prediction.get("confidence", 0.5)
    risk_score = prediction.get("risk_score", 50)
    model_name = prediction.get("model_name", request.model_type.value)

    # Calculate processing time
    processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

    # Build analysis and explanation
    analysis = _build_analysis_breakdown(features, individual_predictions)
    explanation = _generate_explanation(verdict_str, model_name)

    # Build response
    response = AnalysisResponse(
        email_id=email_id,
        verdict=Verdict(verdict_str),
        confidence=float(confidence),
        risk_score=risk_score,
        risk_level=_risk_score_to_level(risk_score),
        model_used=model_name,
        individual_predictions=individual_predictions if len(individual_predictions) > 1 else None,
        analysis=analysis,
        explanation=explanation,
        processing_time_ms=processing_time_ms,
        cache_hit=False,
        timestamp=datetime.utcnow().isoformat()
    )

    # Cache result
    if request.use_cache and request.model_type != ModelType.ENSEMBLE:
        features_str = json.dumps(features, sort_keys=True)
        cache_key = f"prediction:{request.model_type.value}:{hashlib.md5(features_str.encode()).hexdigest()}"
        await cache_service.set(
            cache_key,
            response.model_dump(),
            ttl=settings.REDIS_PREDICTION_CACHE_TTL
        )

    # Record metrics
    record_model_prediction(
        model_type=request.model_type.value,
        verdict=verdict_str,
        duration_ms=processing_time_ms
    )

    logger.info(f"Email analyzed: {verdict_str} (confidence: {confidence:.2f}, risk: {risk_score})")

    return response


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple emails in batch.

    Maximum 100 emails per batch. Supports parallel processing.

    **Note**: Currently only URL analysis is fully implemented.
    Email analysis requires ML models (Phase 3).
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    start_time = datetime.now()

    try:
        # Validate batch size
        if len(request.emails) > 100:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of 100. Got {len(request.emails)} emails"
            )

        # Process emails in parallel if requested
        if request.parallel:
            tasks = []
            for email_req in request.emails:
                # Check if this is a URL-based request (dict with 'url' key)
                if isinstance(email_req, dict) and 'url' in email_req:
                    tasks.append(analyze_url(
                        URLAnalysisRequest(
                            url=email_req.get('url'),
                            context=email_req.get('context'),
                            use_cache=email_req.get('use_cache', True)
                        )
                    ))
                # Skip email analysis for now
                else:
                    logger.warning(f"Batch email analysis not yet supported, skipping: {type(email_req)}")

            results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        else:
            results = []
            for email_req in request.emails:
                if isinstance(email_req, dict) and 'url' in email_req:
                    try:
                        result = await analyze_url(
                            URLAnalysisRequest(
                                url=email_req.get('url'),
                                context=email_req.get('context'),
                                use_cache=email_req.get('use_cache', True)
                            )
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch item failed: {e}")
                else:
                    logger.warning(f"Batch email analysis not yet supported, skipping: {type(email_req)}")

        successful = [r for r in results if isinstance(r, AnalysisResponse)]
        failed = len(results) - len(successful)

        # Calculate summary statistics
        if successful:
            phishing_count = sum(1 for r in successful if r.verdict == Verdict.PHISHING)
            legitimate_count = sum(1 for r in successful if r.verdict == Verdict.LEGITIMATE)
            suspicious_count = sum(1 for r in successful if r.verdict == Verdict.SUSPICIOUS)
            avg_risk_score = sum(r.risk_score for r in successful) / len(successful)
            avg_confidence = sum(r.confidence for r in successful) / len(successful)
        else:
            phishing_count = legitimate_count = suspicious_count = 0
            avg_risk_score = avg_confidence = 0.0

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return BatchAnalysisResponse(
            batch_id=batch_id,
            results=successful,
            summary={
                "total_emails": len(request.emails),
                "phishing_count": phishing_count,
                "legitimate_count": legitimate_count,
                "suspicious_count": suspicious_count,
                "avg_risk_score": round(avg_risk_score, 2),
                "avg_confidence": round(avg_confidence, 2)
            },
            total_processing_time_ms=total_time,
            successful_count=len(successful),
            failed_count=failed
        )

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")
