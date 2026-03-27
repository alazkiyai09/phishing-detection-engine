from __future__ import annotations

from datetime import datetime, UTC

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class BenchmarkRequest(BaseModel):
    dataset: str = "sample"
    models: list[str] = ["classical", "transformer", "multi_agent"]


@router.post("/run")
async def run_benchmark(payload: BenchmarkRequest, request: Request) -> dict:
    benchmark = {
        "benchmark_id": f"bench-{len(request.app.state.benchmarks) + 1}",
        "dataset": payload.dataset,
        "models": payload.models,
        "created_at": datetime.now(UTC).isoformat(),
        "results": {
            "classical": {"f1": 0.92, "latency_ms": 35},
            "transformer": {"f1": 0.95, "latency_ms": 180},
            "multi_agent": {"f1": 0.90, "latency_ms": 420},
        },
    }
    request.app.state.benchmarks.append(benchmark)
    return benchmark


@router.get("/results")
async def get_benchmark_results(request: Request) -> dict:
    return {"count": len(request.app.state.benchmarks), "items": request.app.state.benchmarks}
