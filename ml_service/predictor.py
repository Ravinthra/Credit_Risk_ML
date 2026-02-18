"""
Credit Risk ML — ML Predictor for FastAPI
==========================================
Encapsulates model loading, preprocessing, prediction, and explanation.

WHY A SEPARATE PREDICTOR CLASS?
───────────────────────────────
- FastAPI routes stay thin (just HTTP logic)
- Predictor can be unit tested without HTTP
- Model loading happens once at startup, not per-request
- Same pattern used at scale: Uber, Airbnb, Stripe all separate
  "ML inference engine" from "API handler"
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path so we can import from src/
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_registry import ModelRegistry, ModelVersion
from webapp.predictor.ml.preprocessor import InferencePreprocessor
from schemas import LoanApplication, PredictionResponse, ShapFeature


class MLPredictor:
    """
    ML prediction engine for the FastAPI microservice.

    Loads model from registry, preprocesses inputs, runs inference,
    and generates SHAP explanations.

    Usage:
        predictor = MLPredictor()
        predictor.load()
        response = predictor.predict(loan_application)
    """

    def __init__(self, registry_dir: Path = None):
        self._registry = ModelRegistry.get_instance(registry_dir)
        self._model_version: ModelVersion | None = None
        self._preprocessor = InferencePreprocessor()
        self._explainer = None
        self._loaded = False

    def load(self, version: str = None):
        """
        Load model from registry.

        Parameters
        ----------
        version : str, optional
            Specific version to load (e.g., "v1"). 
            Defaults to latest version.
        """
        if version:
            self._model_version = self._registry.load_model_version(version)
        else:
            self._model_version = self._registry.load_latest_model()

        # Initialize SHAP explainer
        try:
            from webapp.predictor.ml.explainer import ShapExplainer
            self._explainer = ShapExplainer()
        except ImportError:
            print("  [Warning] SHAP not available, explanations disabled.")
            self._explainer = None

        self._loaded = True
        meta = self._model_version.metadata
        print(f"  [Predictor] Loaded {meta.get('model_name', '?')} "
              f"{meta.get('version', '?')} "
              f"(AUC: {meta.get('auc_roc', 0):.4f})")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        if self._model_version:
            return self._model_version.version
        return ""

    @property
    def model_name(self) -> str:
        if self._model_version:
            return self._model_version.metadata.get('model_name', 'unknown')
        return "unknown"

    @property
    def metadata(self) -> dict:
        if self._model_version:
            return self._model_version.metadata
        return {}

    def predict(self, loan: LoanApplication) -> PredictionResponse:
        """
        Run full prediction pipeline.

        1. Convert Pydantic model → dict
        2. Preprocess → 66-feature DataFrame
        3. Predict probability
        4. Classify risk category
        5. Compute SHAP explanations
        6. Compute EMI

        Parameters
        ----------
        loan : LoanApplication
            Validated loan application data.

        Returns
        -------
        PredictionResponse with all fields populated.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        model = self._model_version.model
        meta = self._model_version.metadata

        # 1. Preprocess
        form_data = loan.model_dump()
        features_df = self._preprocessor.preprocess(form_data)

        # 2. Predict
        proba = float(model.predict_proba(features_df)[:, 1][0])

        # 3. Risk category (using optimal threshold from metadata)
        optimal_threshold = meta.get('optimal_threshold', 0.5)
        risk_info = self._get_risk_category(proba, optimal_threshold)

        # 4. SHAP explanations
        shap_features = []
        if self._explainer:
            try:
                explanations = self._explainer.explain(model, features_df, top_n=5)
                shap_features = [
                    ShapFeature(
                        feature=exp.get('feature', ''),
                        impact=exp.get('impact_pct', 0),
                        direction=exp.get('direction', 'unknown'),
                    )
                    for exp in explanations[:5]
                ]
            except Exception as e:
                print(f"  [Warning] SHAP failed: {e}")

        # 5. EMI
        emi = InferencePreprocessor._compute_emi(
            loan.loan_amount, loan.interest_rate, loan.loan_term_months
        )

        return PredictionResponse(
            probability=round(proba * 100, 2),
            probability_raw=round(proba, 6),
            risk_category=risk_info['category'],
            risk_color=risk_info['color'],
            description=risk_info['description'],
            recommended_action=risk_info['action'],
            optimal_threshold_used=optimal_threshold,
            model_version=self._model_version.version,
            model_name=meta.get('model_name', 'unknown'),
            emi=round(emi, 2),
            top_5_shap_features=shap_features,
        )

    @staticmethod
    def _get_risk_category(probability: float, threshold: float = 0.5) -> dict:
        """
        Convert probability to risk category.

        Uses the business-optimized threshold from the registry
        instead of hardcoded 0.5.
        """
        if probability < threshold * 0.4:
            return {
                'category': 'Low Risk',
                'color': '#10b981',
                'description': 'Strong repayment profile. Standard approval recommended.',
                'action': 'Auto-Approve',
            }
        elif probability < threshold:
            return {
                'category': 'Medium Risk',
                'color': '#f59e0b',
                'description': 'Some risk indicators present. Manual review suggested.',
                'action': 'Manual Review',
            }
        else:
            return {
                'category': 'High Risk',
                'color': '#ef4444',
                'description': 'Significant default risk. Additional verification required.',
                'action': 'Decline / Escalate',
            }
