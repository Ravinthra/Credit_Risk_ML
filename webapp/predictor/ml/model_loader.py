"""
Credit Risk ML -- Model Loader (Singleton Pattern)
===================================================
Loads trained ML models from pickle files and caches them in memory.

WHY SINGLETON PATTERN?
- Loading a model from disk takes ~100ms
- We don't want to reload on every request
- Thread-safe singleton ensures one load across all Django workers
- In production, each gunicorn worker loads its own copy (process-level)

WHY SEPARATE FROM VIEWS?
- Views should not know HOW models are loaded
- This module can be unit tested independently
- Swapping model format (pkl -> ONNX) only changes this file

Interview Insight:
"How do you serve ML models in production?"
-> "I use a singleton loader that deserializes the model once at startup
   and caches it in memory. For high-throughput, I'd use ONNX Runtime
   or TensorFlow Serving, but for Django monoliths with moderate traffic,
   joblib + in-memory caching is the pragmatic choice."
"""

import json
import sys
import threading
import joblib
import pandas as pd
from pathlib import Path
from django.conf import settings

# Model .pkl files were pickled with src/ on sys.path, so pickle's
# find_class needs to resolve the 'preprocessing' module at load time.
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class ModelLoader:
    """
    Thread-safe singleton model loader.

    Usage:
        loader = ModelLoader.get_instance()
        proba = loader.predict_proba(features_df)
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        # WHY not load in __init__?
        # We use lazy loading -- model is loaded on first predict call.
        # This avoids slowing down Django startup for management commands
        # like migrate, collectstatic, etc.
        self._models = {}
        self._default_model = None
        self._training_report = None
        self._model_comparison = None
        self._scaler = None
        self._scaler_loaded = False

    # Models that were trained on scaled features
    SCALED_MODELS = {'Logistic_Regression'}

    @classmethod
    def get_instance(cls):
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_model(self, model_name: str):
        """Load a single model from disk."""
        model_path = settings.ML_CONFIG['available_models'].get(model_name)
        if model_path is None:
            raise ValueError(f"Unknown model: {model_name}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"  [ML] Loading {model_name} from {model_path}...")
        model = joblib.load(model_path)
        self._models[model_name] = model
        return model

    def get_model(self, model_name: str = None):
        """
        Get a loaded model by name. Loads from disk if not cached.

        Parameters
        ----------
        model_name : str, optional
            Model to load. Defaults to settings.ML_CONFIG['default_model'].

        Returns
        -------
        Trained sklearn/xgboost/lightgbm model object.
        """
        if model_name is None:
            model_name = settings.ML_CONFIG['default_model']

        if model_name not in self._models:
            self._load_model(model_name)

        return self._models[model_name]

    def predict_proba(self, features_df: pd.DataFrame, model_name: str = None) -> float:
        """
        Get default probability for a single loan application.

        Parameters
        ----------
        features_df : pd.DataFrame
            Single-row DataFrame with all 66 features (preprocessed).
        model_name : str, optional
            Which model to use. Defaults to the configured default.

        Returns
        -------
        float: Probability of default (0.0 to 1.0)
        """
        if model_name is None:
            model_name = settings.ML_CONFIG['default_model']
        model = self.get_model(model_name)

        # Apply scaling for models trained on scaled features
        if model_name in self.SCALED_MODELS:
            features_df = self._apply_scaling(features_df)

        proba = model.predict_proba(features_df)[:, 1]
        return float(proba[0])

    def _apply_scaling(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature scaling using the saved scaler from training.

        WHY?
        Logistic Regression was trained on StandardScaler/RobustScaler-
        transformed features. Without scaling at inference, predictions
        are essentially 0.0 for all inputs.
        """
        if not self._scaler_loaded:
            scaler_path = settings.ML_MODELS_DIR / 'feature_scaler.pkl'
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
                print(f"  [ML] Loaded feature scaler from {scaler_path}")
            else:
                print(f"  [ML] WARNING: Scaler not found at {scaler_path}")
            self._scaler_loaded = True

        if self._scaler is not None:
            return self._scaler.transform(features_df, apply_scaling=True)
        return features_df

    def get_risk_category(self, probability: float) -> dict:
        """
        Convert probability to risk category with color coding.

        WHY THESE THRESHOLDS?
        - <20%: Low risk -- standard approval
        - 20-50%: Medium risk -- manual review / higher rate
        - >50%: High risk -- likely rejection

        In production, thresholds are calibrated using business cost:
        cost_of_missed_default vs cost_of_rejected_good_customer
        """
        if probability < 0.20:
            return {
                'category': 'Low Risk',
                'color': '#10b981',     # Green
                'description': 'Strong repayment profile. Standard approval recommended.',
                'action': 'Auto-Approve',
            }
        elif probability < 0.50:
            return {
                'category': 'Medium Risk',
                'color': '#f59e0b',     # Amber
                'description': 'Some risk indicators present. Manual review suggested.',
                'action': 'Manual Review',
            }
        else:
            return {
                'category': 'High Risk',
                'color': '#ef4444',     # Red
                'description': 'Significant default risk. Additional verification required.',
                'action': 'Decline / Escalate',
            }

    def get_training_report(self) -> dict:
        """Load the training report JSON for the dashboard."""
        if self._training_report is None:
            report_path = settings.ML_CONFIG['training_report']
            if report_path.exists():
                with open(report_path, 'r') as f:
                    self._training_report = json.load(f)
            else:
                self._training_report = {}
        return self._training_report

    def get_model_comparison(self) -> list:
        """Load model comparison CSV for the dashboard."""
        if self._model_comparison is None:
            comp_path = settings.ML_CONFIG['model_comparison']
            if comp_path.exists():
                df = pd.read_csv(comp_path)
                self._model_comparison = df.to_dict(orient='records')
            else:
                self._model_comparison = []
        return self._model_comparison
