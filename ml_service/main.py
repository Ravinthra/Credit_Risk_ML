"""
Credit Risk ML — FastAPI Microservice
=======================================
Stateless ML inference service for credit risk prediction.

WHY FASTAPI INSTEAD OF DJANGO REST?
────────────────────────────────────
- 3-5x faster than Django REST (async + Starlette)
- Automatic OpenAPI/Swagger docs at /docs
- Pydantic validation built-in (no serializer classes)
- Designed for microservices, not monoliths

In production architecture:
  Django handles UI + admin + sessions
  FastAPI handles ML inference (stateless, horizontally scalable)
  Both share the model registry via mounted volume

Run:
    uvicorn ml_service.main:app --host 0.0.0.0 --port 8001 --reload

Docs:
    http://localhost:8001/docs
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add ml_service to path for local imports
ML_SERVICE_DIR = Path(__file__).parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    LoanApplication, PredictionResponse,
    HealthResponse, ModelInfoResponse,
)
from predictor import MLPredictor

# ─── Global predictor instance ──────────────────────────
ml_predictor = MLPredictor()


# ─── Lifespan (startup/shutdown) ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model at startup, cleanup at shutdown."""
    print("\n  [FastAPI] Starting Credit Risk ML Service...")
    try:
        ml_predictor.load()
        print("  [FastAPI] Model loaded successfully.\n")
    except Exception as e:
        print(f"  [FastAPI] WARNING: Model load failed: {e}")
        print("  [FastAPI] Service will start but /predict will fail.\n")
    yield
    print("\n  [FastAPI] Shutting down ML Service.")


# ─── App ────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk ML API",
    description=(
        "Production-grade ML inference API for loan default prediction. "
        "Uses LightGBM with SHAP explainability and business-optimized thresholds."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow Django frontend to call this service)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/predict", response_model=PredictionResponse,
          summary="Predict loan default risk",
          tags=["Prediction"])
async def predict(loan: LoanApplication):
    """
    Predict credit risk for a loan application.

    Accepts applicant details and returns:
    - Default probability
    - Risk category (Low/Medium/High)
    - Business-optimized threshold used
    - Model version for audit trail
    - Top 5 SHAP feature explanations
    - Estimated EMI

    Example request body:
    ```json
    {
        "age": 28,
        "gender": "Male",
        "education": "Bachelor",
        "marital_status": "Single",
        "dependents": 0,
        "employment_type": "Salaried",
        "employment_years": 5,
        "annual_income": 600000,
        "credit_score": 720,
        "credit_utilization": 40,
        "existing_loans": 1,
        "loan_type": "Personal",
        "loan_amount": 500000,
        "loan_term_months": 36,
        "interest_rate": 12.0
    }
    ```
    """
    if not ml_predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Service is starting up."
        )

    try:
        result = ml_predictor.predict(loan)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse,
         summary="Service health check",
         tags=["Operations"])
async def health():
    """Check if the service is healthy and model is loaded."""
    return HealthResponse(
        status="healthy" if ml_predictor.is_loaded else "degraded",
        model_loaded=ml_predictor.is_loaded,
        model_version=ml_predictor.model_version,
        model_name=ml_predictor.model_name,
    )


@app.get("/model-info", response_model=ModelInfoResponse,
         summary="Get loaded model information",
         tags=["Operations"])
async def model_info():
    """Get details about the currently loaded model and available versions."""
    from src.model_registry import ModelRegistry

    registry = ModelRegistry.get_instance()
    meta = ml_predictor.metadata
    versions = registry.list_versions()

    return ModelInfoResponse(
        version=meta.get('version', ''),
        model_name=meta.get('model_name', ''),
        auc_roc=meta.get('auc_roc', 0),
        feature_count=meta.get('feature_count', 0),
        training_date=meta.get('training_date', ''),
        optimal_threshold=meta.get('optimal_threshold', 0.5),
        available_versions=versions,
    )
