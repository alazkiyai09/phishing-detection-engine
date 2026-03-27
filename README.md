<div align="center">

# 📧 Phishing Detection Engine

### URL & Email Analysis • Ensemble ML • Transformer • Multi-Agent Explainability

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FCC624?style=flat)](https://huggingface.co/docs/transformers)
[![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=flat)](https://lightgbm.readthedocs.io/)

[Overview](#-overview) • [About](#-about) • [Topics](#-topics) • [API](#-api-surfaces) • [Quick Start](#-quick-start)

---

Phishing detection platform combining **heuristics**, **classical ML**, **transformers**, and **multi-agent analysis** with explainable API outputs.

</div>

---

## 🎯 Overview

`phishing-detection-engine` delivers layered phishing analysis:

- URL, header, and content feature extraction
- Classical and transformer model inference paths
- Multi-agent decision support and explanation flow
- Unified API for analyze, explain, and benchmarking

## 📌 About

- Built for fast phishing triage and model experimentation
- Supports both lightweight and deep-analysis paths
- Includes explainability outputs suitable for analyst workflows

## 🏷️ Topics

`phishing-detection` `email-security` `url-analysis` `machine-learning` `transformers` `fastapi` `cybersecurity` `explainable-ai`

## 🧩 Architecture

- `src/features/`: feature extraction pipelines
- `src/models/`: classical, transformer, multi-agent, ensemble
- `src/explainability/`: narrative and contribution mapping
- `src/api/`: unified API + compatibility app
- `src/core/`: errors, logging, validation, security

## 🌐 API Surfaces

- `POST /api/v1/analyze/url`
- `POST /api/v1/analyze/email`
- `POST /api/v1/analyze/batch`
- `POST /api/v1/explain/{id}`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/benchmark/results`
- `GET /health`
- `GET /metrics`

## ⚡ Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## 🔧 GLM Setup

```bash
export GLM_API_KEY=your_glm_api_key
export GLM_BASE_URL=https://api.z.ai/api/anthropic
export GLM_MODEL=glm-5.1
```

## 🛠️ Tech Stack

**ML:** scikit-learn, XGBoost, LightGBM, Transformers  
**API:** FastAPI, Pydantic  
**XAI:** SHAP/LIME narrative layers  
**Ops:** Redis, Prometheus-ready routing
