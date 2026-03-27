# phishing-detection-engine

ML engine for phishing detection with feature extraction, classical models, transformer wrappers, multi-agent analysis, explainability, and a unified FastAPI surface.

## Layout

- `src/features/`: feature extraction pipeline and heuristics
- `src/models/`: classical, transformer, multi-agent, and ensemble model surfaces
- `src/explainability/`: explanation helpers and legacy XAI assets
- `src/api/`: unified FastAPI app and legacy API copy
- `src/core/`: embedded shared utilities copied from `shared_utils/`
- `tests/`: lightweight smoke tests plus preserved legacy test trees

## Run

```bash
uvicorn src.api.main:app --reload
```

## Notes

- The repo preserves the original project trees under nested namespaces and adds a thin integration layer on top.
- Legacy test suites are retained for reference; `tests/test_api.py` is the lightweight smoke test for the unified API shell.
