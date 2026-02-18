"""
Credit Risk ML — Model Registry & Versioning
==============================================
Production-grade model lifecycle management with versioned storage.

WHY MODEL VERSIONING?
─────────────────────
In production, you never deploy "a model" — you deploy "model v7 trained
on 2026-02-01 data." Every prediction must be traceable to:
  - Exact model weights
  - Training data snapshot
  - Preprocessing artifacts (scaler, encoders)
  - Hyperparameters and metrics

This enables:
  1. Rollback: If v8 degrades, instantly revert to v7
  2. A/B testing: Route 10% traffic to v8, compare
  3. Audit: Regulator asks "why did you reject this loan?" → load exact model

Interview Insight:
"How do you manage model versions in production?"
→ "I built a file-based registry where each version stores model + scaler +
   encoders + metadata atomically. In larger orgs, I'd use MLflow or
   Vertex AI Model Registry, but the concept is identical: immutable
   versioned snapshots with full lineage."
"""

import json
import shutil
import threading
import platform
import sys
import joblib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ─── Paths ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"


# ============================================================
# 1. MODEL VERSION DATACLASS
# ============================================================

@dataclass
class ModelVersion:
    """
    Immutable snapshot of a trained model and its full context.

    WHY ALL THESE FIELDS?
    ─────────────────────
    - model: The trained estimator (predict/predict_proba)
    - scaler: Exact scaler fitted on training data (prevents skew)
    - encoders: Categorical encoders (label, ordinal, target)
    - metadata: Everything else (metrics, params, dates, versions)

    In production, this is what gets loaded at serve-time.
    """
    model: Any = None
    scaler: Any = None
    encoders: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    version: str = ""


@dataclass
class ModelMetadata:
    """Structured metadata for a model version."""
    version: str = ""
    model_name: str = ""
    model_class: str = ""
    training_date: str = ""
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    f1_score: float = 0.0
    ks_statistic: float = 0.0
    gini: float = 0.0
    feature_count: int = 0
    hyperparameters: dict = field(default_factory=dict)
    dataset_size: dict = field(default_factory=dict)
    optimal_threshold: float = 0.5
    python_version: str = ""
    dependencies: dict = field(default_factory=dict)
    notes: str = ""


# ============================================================
# 2. MODEL REGISTRY
# ============================================================

class ModelRegistry:
    """
    Thread-safe model registry with versioned storage.

    Folder Structure:
        registry/
          v1/
            model.pkl
            scaler.pkl
            encoders.pkl
            metadata.json
          v2/
            ...

    Usage:
        registry = ModelRegistry.get_instance()
        registry.save_model_version(model, scaler, encoders, metrics, ...)
        latest = registry.load_latest_model()
        meta = registry.get_model_metadata()
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, registry_dir: Path = None):
        self._registry_dir = registry_dir or DEFAULT_REGISTRY_DIR
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ModelVersion] = {}

    @classmethod
    def get_instance(cls, registry_dir: Path = None) -> 'ModelRegistry':
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(registry_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # ─── Version Management ───────────────────────────────

    def _get_all_versions(self) -> list[str]:
        """Get sorted list of version directories (v1, v2, ...)."""
        versions = []
        if self._registry_dir.exists():
            for d in self._registry_dir.iterdir():
                if d.is_dir() and d.name.startswith('v'):
                    try:
                        int(d.name[1:])  # Validate format
                        versions.append(d.name)
                    except ValueError:
                        continue
        return sorted(versions, key=lambda v: int(v[1:]))

    def _next_version(self) -> str:
        """Generate next version string."""
        versions = self._get_all_versions()
        if not versions:
            return "v1"
        latest_num = int(versions[-1][1:])
        return f"v{latest_num + 1}"

    def _latest_version(self) -> Optional[str]:
        """Get the latest version string, or None if empty."""
        versions = self._get_all_versions()
        return versions[-1] if versions else None

    # ─── Save ──────────────────────────────────────────────

    def save_model_version(
        self,
        model: Any,
        scaler: Any = None,
        encoders: dict = None,
        metrics: dict = None,
        hyperparameters: dict = None,
        dataset_info: dict = None,
        model_name: str = "",
        optimal_threshold: float = 0.5,
        notes: str = "",
    ) -> str:
        """
        Save a new model version to the registry.

        Parameters
        ----------
        model : trained model object
        scaler : fitted scaler (StandardScaler, etc.)
        encoders : dict of fitted encoders {name: encoder}
        metrics : dict with auc_roc, f1_score, etc.
        hyperparameters : dict of model hyperparameters
        dataset_info : dict with train_size, test_size, feature_count
        model_name : str, e.g. "LightGBM"
        optimal_threshold : float, business-optimized threshold
        notes : str, free-text notes

        Returns
        -------
        str: Version string (e.g., "v3")
        """
        with self._lock:
            version = self._next_version()
            version_dir = self._registry_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)

            metrics = metrics or {}
            hyperparameters = hyperparameters or {}
            dataset_info = dataset_info or {}
            encoders = encoders or {}

            # Serialize artifacts
            joblib.dump(model, version_dir / 'model.pkl')
            print(f"  [Registry] Saved model -> {version_dir / 'model.pkl'}")

            if scaler is not None:
                joblib.dump(scaler, version_dir / 'scaler.pkl')

            if encoders:
                joblib.dump(encoders, version_dir / 'encoders.pkl')

            # Build metadata
            metadata = ModelMetadata(
                version=version,
                model_name=model_name or type(model).__name__,
                model_class=f"{type(model).__module__}.{type(model).__name__}",
                training_date=datetime.now().isoformat(),
                auc_roc=metrics.get('auc_roc', 0),
                auc_pr=metrics.get('auc_pr', 0),
                f1_score=metrics.get('f1_score', 0),
                ks_statistic=metrics.get('ks_statistic', 0),
                gini=metrics.get('gini', 0),
                feature_count=dataset_info.get('feature_count', 0),
                hyperparameters=self._serialize_params(hyperparameters),
                dataset_size={
                    'train': dataset_info.get('train_size', 0),
                    'test': dataset_info.get('test_size', 0),
                },
                optimal_threshold=optimal_threshold,
                python_version=platform.python_version(),
                dependencies=self._get_dependencies(),
                notes=notes,
            )

            # Write metadata JSON
            meta_path = version_dir / 'metadata.json'
            with open(meta_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)

            print(f"  [Registry] Version {version} saved successfully.")
            return version

    # ─── Load ──────────────────────────────────────────────

    def load_model_version(self, version: str) -> ModelVersion:
        """
        Load a specific model version from disk.

        Parameters
        ----------
        version : str, e.g. "v1", "v3"

        Returns
        -------
        ModelVersion with all artifacts loaded.
        """
        # Check cache first
        if version in self._cache:
            return self._cache[version]

        version_dir = self._registry_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version} not found in registry.")

        # Load artifacts
        model = joblib.load(version_dir / 'model.pkl')

        scaler = None
        scaler_path = version_dir / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        encoders = {}
        encoders_path = version_dir / 'encoders.pkl'
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)

        metadata = {}
        meta_path = version_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        mv = ModelVersion(
            model=model,
            scaler=scaler,
            encoders=encoders,
            metadata=metadata,
            version=version,
        )

        # Cache it
        self._cache[version] = mv
        print(f"  [Registry] Loaded {version}: {metadata.get('model_name', 'unknown')}")
        return mv

    def load_latest_model(self) -> ModelVersion:
        """
        Load the most recent model version.

        Returns
        -------
        ModelVersion of the latest registered version.

        Raises
        ------
        FileNotFoundError if registry is empty.
        """
        latest = self._latest_version()
        if latest is None:
            raise FileNotFoundError("Registry is empty. No models registered.")
        return self.load_model_version(latest)

    # ─── Metadata ──────────────────────────────────────────

    def get_model_metadata(self, version: str = None) -> dict:
        """
        Get metadata for a specific version (or latest).

        Parameters
        ----------
        version : str, optional. Defaults to latest version.

        Returns
        -------
        dict with full metadata.
        """
        if version is None:
            version = self._latest_version()
            if version is None:
                return {}

        meta_path = self._registry_dir / version / 'metadata.json'
        if not meta_path.exists():
            return {}

        with open(meta_path, 'r') as f:
            return json.load(f)

    def list_versions(self) -> list[dict]:
        """
        List all registered versions with summary info.

        Returns
        -------
        List of dicts with version, model_name, auc_roc, training_date.
        """
        summaries = []
        for version in self._get_all_versions():
            meta = self.get_model_metadata(version)
            summaries.append({
                'version': version,
                'model_name': meta.get('model_name', 'unknown'),
                'auc_roc': meta.get('auc_roc', 0),
                'f1_score': meta.get('f1_score', 0),
                'training_date': meta.get('training_date', ''),
                'feature_count': meta.get('feature_count', 0),
                'dataset_size': meta.get('dataset_size', {}),
            })
        return summaries

    def delete_version(self, version: str):
        """Delete a model version (irreversible)."""
        version_dir = self._registry_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            self._cache.pop(version, None)
            print(f"  [Registry] Deleted {version}")
        else:
            print(f"  [Registry] Version {version} not found.")

    # ─── Helpers ───────────────────────────────────────────

    @staticmethod
    def _serialize_params(params: dict) -> dict:
        """Convert numpy types to JSON-serializable Python types."""
        import numpy as np
        clean = {}
        for k, v in params.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        return clean

    @staticmethod
    def _get_dependencies() -> dict:
        """Capture key package versions for reproducibility."""
        deps = {}
        for pkg in ['sklearn', 'xgboost', 'lightgbm', 'pandas', 'numpy', 'joblib']:
            try:
                mod = __import__(pkg)
                deps[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                pass
        return deps

    def print_registry(self):
        """Print formatted registry summary."""
        versions = self.list_versions()
        if not versions:
            print("\n  Registry is empty.")
            return

        print(f"\n  {'─' * 75}")
        print(f"  {'Version':<10} {'Model':<20} {'AUC-ROC':>10} {'F1':>10} {'Features':>10} {'Date':>15}")
        print(f"  {'─' * 75}")
        for v in versions:
            date = v['training_date'][:10] if v['training_date'] else ''
            print(f"  {v['version']:<10} {v['model_name']:<20} {v['auc_roc']:>10.4f} "
                  f"{v['f1_score']:>10.4f} {v['feature_count']:>10} {date:>15}")
        print(f"  {'─' * 75}")


# ============================================================
# 3. CONVENIENCE: REGISTER CURRENT MODELS
# ============================================================

def register_existing_models():
    """
    Register the 4 existing trained models as v1, v2, v3, v4.
    
    Reads metrics from training_report.json and saves each model
    into the registry with full metadata.
    """
    import pandas as pd

    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    DATA_DIR = PROJECT_ROOT / "data" / "processed"

    # Load training report
    with open(RESULTS_DIR / 'training_report.json', 'r') as f:
        report = json.load(f)

    # Load dataset info
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')

    # Load threshold optimization results if available
    optimal_threshold = 0.5
    thresh_path = RESULTS_DIR / 'threshold_optimization.json'
    if thresh_path.exists():
        with open(thresh_path, 'r') as f:
            thresh_data = json.load(f)
            optimal_threshold = thresh_data.get('optimal_threshold', {}).get('threshold', 0.5)

    registry = ModelRegistry.get_instance()

    # Register each model from the report
    model_order = ['LightGBM', 'XGBoost', 'Logistic_Regression', 'Sklearn_GBM']

    for model_name in model_order:
        model_path = MODELS_DIR / f'{model_name}_model.pkl'
        if not model_path.exists():
            print(f"  [Skip] {model_name} not found at {model_path}")
            continue

        model = joblib.load(model_path)
        details = report.get('model_details', {}).get(model_name, {})
        metrics = details.get('metrics', {})
        hyperparams = details.get('best_params', {})

        version = registry.save_model_version(
            model=model,
            scaler=None,     # Scaler was applied during preprocessing
            encoders={},     # Encoders were applied during preprocessing
            metrics=metrics,
            hyperparameters=hyperparams,
            dataset_info={
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': X_train.shape[1],
            },
            model_name=model_name,
            optimal_threshold=optimal_threshold if model_name == 'LightGBM' else 0.5,
            notes=f"Registered from existing training run ({report.get('timestamp', '')})",
        )
        print(f"  -> {model_name} registered as {version}")

    print()
    registry.print_registry()


if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  MODEL REGISTRY — Registering Existing Models")
    print("=" * 65 + "\n")
    register_existing_models()
    print("\n  Done.\n")
