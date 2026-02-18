"""
Credit Risk ML — Pydantic Schemas for FastAPI
===============================================
Request/response validation models.

WHY PYDANTIC?
─────────────
- Automatic request validation with clear error messages
- Type coercion (string "28" → int 28)
- OpenAPI/Swagger docs generated automatically
- Prevents malformed inputs from reaching the model
"""

from pydantic import BaseModel, Field
from typing import Optional


class LoanApplication(BaseModel):
    """
    Input schema for credit risk prediction.

    Each field maps to a form input in the Django frontend.
    Constraints match the Django form's min/max validators.
    """
    age: int = Field(..., ge=18, le=70, description="Applicant age (18-70)")
    gender: str = Field(..., description="Male / Female / Other")
    education: str = Field(..., description="Below Secondary / Secondary / Higher Secondary / Bachelor / Master / PhD")
    marital_status: str = Field(..., description="Single / Married / Divorced / Widowed")
    dependents: int = Field(0, ge=0, le=10, description="Number of dependents")
    employment_type: str = Field(..., description="Salaried / Self-Employed / Business / Freelance / Unemployed")
    employment_years: float = Field(..., ge=0, le=40, description="Years of employment")
    annual_income: float = Field(..., ge=50000, le=50000000, description="Annual income in INR")
    credit_score: int = Field(..., ge=300, le=900, description="Credit bureau score")
    credit_utilization: float = Field(40, ge=0, le=100, description="Credit utilization percentage")
    existing_loans: int = Field(1, ge=0, le=20, description="Number of existing loans")
    loan_type: str = Field(..., description="Personal / Home / Auto / Education / Business / Gold / Credit Card")
    loan_amount: float = Field(..., ge=10000, le=50000000, description="Requested loan amount in INR")
    loan_term_months: int = Field(36, ge=3, le=360, description="Loan tenure in months")
    interest_rate: float = Field(12.0, ge=1, le=40, description="Annual interest rate (%)")

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                    "interest_rate": 12.0,
                }
            ]
        }
    }


class ShapFeature(BaseModel):
    """Single SHAP feature contribution."""
    feature: str = Field(..., description="Human-readable feature name")
    impact: float = Field(..., description="SHAP impact value")
    direction: str = Field(..., description="increases_risk / decreases_risk")


class PredictionResponse(BaseModel):
    """
    Output schema for credit risk prediction.

    Includes probability, business decision, model provenance,
    and explainability — everything a downstream consumer needs.
    """
    probability: float = Field(..., description="Default probability (0-100%)")
    probability_raw: float = Field(..., description="Raw probability (0-1)")
    risk_category: str = Field(..., description="Low Risk / Medium Risk / High Risk")
    risk_color: str = Field(..., description="Color code for UI (#10b981, #f59e0b, #ef4444)")
    description: str = Field(..., description="Human-readable risk description")
    recommended_action: str = Field(..., description="Auto-Approve / Manual Review / Decline")
    optimal_threshold_used: float = Field(..., description="Business-optimized threshold")
    model_version: str = Field(..., description="Registry version (e.g., v1)")
    model_name: str = Field(..., description="Model name (e.g., LightGBM)")
    emi: float = Field(..., description="Estimated monthly installment (INR)")
    top_5_shap_features: list[ShapFeature] = Field(
        default_factory=list,
        description="Top 5 SHAP feature contributions"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    model_version: str = ""
    model_name: str = ""


class ModelInfoResponse(BaseModel):
    """Model information response."""
    version: str = ""
    model_name: str = ""
    auc_roc: float = 0.0
    feature_count: int = 0
    training_date: str = ""
    optimal_threshold: float = 0.5
    available_versions: list[dict] = Field(default_factory=list)
