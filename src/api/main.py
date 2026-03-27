from fastapi import FastAPI

from src.api.routers import analyze, benchmark, explain


app = FastAPI(
    title="phishing-detection-engine",
    version="0.1.0",
    description="Unified phishing detection API across feature, model, and explainability layers.",
)

app.include_router(analyze.router, prefix="/api/v1/analyze", tags=["analyze"])
app.include_router(explain.router, prefix="/api/v1/explain", tags=["explain"])
app.include_router(benchmark.router, prefix="/api/v1/benchmark", tags=["benchmark"])


@app.on_event("startup")
async def startup() -> None:
    app.state.predictions = {}
    app.state.benchmarks = []


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "phishing-detection-engine"}
