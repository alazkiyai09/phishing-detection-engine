# Phishing Detection Engine (`phishing-detection-engine`)

Production-focused **phishing detection machine learning engine** combining URL/content/header feature extraction, classical and transformer models, multi-agent analysis, explainability, and API serving.

## Why This Repository

Email and URL threat detection needs layered analysis across heuristics, ML, and language models. `phishing-detection-engine` centralizes those layers in a deployable architecture.

## Core Features

- Feature pipeline for URL, header, content, and linguistic indicators
- Classical ML benchmark and transformer fine-tuning surfaces
- Multi-agent phishing analysis components
- Explainability narrative generation for analyst workflows
- Unified FastAPI endpoints for analyze, explain, and benchmark
- Embedded `src/core` utilities for validation, security, and logging

## Project Structure

- `src/features/`: phishing feature extraction and pipeline composition
- `src/models/`: classical, transformer, multi-agent, and ensemble surfaces
- `src/explainability/`: SHAP extraction and narrative generation layers
- `src/api/`: unified FastAPI app plus compatibility legacy app
- `src/core/`: self-contained shared utilities for the repo

## API Endpoints

- `POST /api/v1/analyze/url`
- `POST /api/v1/analyze/email`
- `POST /api/v1/analyze/batch`
- `POST /api/v1/explain/{id}`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/benchmark/results`
- `GET /health`
- `GET /metrics`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## GLM Configuration

```bash
export GLM_API_KEY=your_glm_api_key
export GLM_BASE_URL=https://api.z.ai/api/anthropic
export GLM_MODEL=glm-5.1
```

## SEO Keywords

phishing detection machine learning, email phishing classifier, url phishing detection api, transformer phishing model, multi agent phishing analysis, explainable phishing ai, fastapi phishing service
