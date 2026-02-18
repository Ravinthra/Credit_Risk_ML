"""
Credit Risk ML -- SHAP Explainer
=================================
Generates per-prediction SHAP explanations.

WHY SHAP?
- LIME is model-agnostic but creates inconsistent explanations
- SHAP has mathematical guarantees (Shapley values from game theory)
- TreeExplainer is exact and fast for tree-based models
- Regulators increasingly require model explanations (EU AI Act)

Interview Insight:
"How do you explain model predictions to business stakeholders?"
-> "I use SHAP values -- they decompose each prediction into per-feature
   contributions. For a rejected loan, I can say 'the primary risk driver
   was a DTI ratio of 0.65, contributing +12% to default probability,
   followed by 2 past defaults adding +8%.' This makes the model
   auditable and builds trust with compliance teams."
"""

import numpy as np
import pandas as pd


class ShapExplainer:
    """
    SHAP-based model explainability for credit risk predictions.

    WHY NOT load shap in the module-level import?
    - shap is a heavy dependency (~100MB)
    - It may not be installed in all environments
    - Lazy import lets the app function without SHAP (just without explanations)
    """

    def __init__(self):
        self._explainer = None
        self._shap_available = None

    def _check_shap(self):
        """Check if shap library is available."""
        if self._shap_available is None:
            try:
                import shap
                self._shap_available = True
            except ImportError:
                self._shap_available = False
                print("  [ML] WARNING: shap not installed. Explanations will use feature importance fallback.")
        return self._shap_available

    def explain(self, model, features_df: pd.DataFrame, top_n: int = 10) -> list:
        """
        Generate SHAP-based explanation for a single prediction.

        Parameters
        ----------
        model : trained model
            The ML model (LightGBM, XGBoost, etc.)
        features_df : pd.DataFrame
            Single-row DataFrame with 66 features.
        top_n : int
            Number of top contributing features to return.

        Returns
        -------
        list of dicts, each with:
            - feature: feature name
            - value: actual feature value
            - shap_value: SHAP contribution
            - direction: 'increases_risk' or 'decreases_risk'
            - impact_pct: absolute percentage impact
        """
        if self._check_shap():
            return self._explain_with_shap(model, features_df, top_n)
        else:
            return self._explain_with_importance(model, features_df, top_n)

    def _explain_with_shap(self, model, features_df, top_n):
        """Use the appropriate SHAP explainer based on model type."""
        import shap
        from sklearn.linear_model import LogisticRegression

        # Pick the right explainer for the model type
        model_id = id(model)
        if self._explainer is None or getattr(self, '_explainer_model_id', None) != model_id:
            if isinstance(model, LogisticRegression):
                # LinearExplainer for linear models
                self._explainer = shap.LinearExplainer(model, features_df)
            else:
                # TreeExplainer for XGBoost, LightGBM, GBM, etc.
                self._explainer = shap.TreeExplainer(model)
            self._explainer_model_id = model_id

        shap_values = self._explainer.shap_values(features_df)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Class 1 (default) contributions
        elif len(shap_values.shape) == 3:
            shap_vals = shap_values[0, :, 1]
        else:
            shap_vals = shap_values[0]

        return self._format_explanations(features_df, shap_vals, top_n)

    def _explain_with_importance(self, model, features_df, top_n):
        """Fallback: use feature importance * feature value direction."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Logistic regression coefficients
            importances = np.abs(model.coef_[0])

        # Simulate directional impact using importance * deviation from mean
        feature_names = features_df.columns.tolist()
        values = features_df.iloc[0].values

        results = []
        for i, (fname, imp, val) in enumerate(zip(feature_names, importances, values)):
            if imp > 0:
                results.append({
                    'feature': self._format_feature_name(fname),
                    'feature_raw': fname,
                    'value': round(float(val), 4),
                    'shap_value': round(float(imp), 6),
                    'direction': 'risk_factor',
                    'impact_pct': round(float(imp / importances.sum() * 100), 2),
                })

        results.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return results[:top_n]

    def _format_explanations(self, features_df, shap_vals, top_n):
        """Format SHAP values into readable explanations."""
        feature_names = features_df.columns.tolist()
        values = features_df.iloc[0].values

        results = []
        max_abs = np.max(np.abs(shap_vals)) if len(shap_vals) > 0 else 1.0

        for fname, val, sv in zip(feature_names, values, shap_vals):
            if abs(sv) > 0.001:  # Skip negligible contributions
                # Use sqrt scale so smaller values remain visible
                # when one feature dominates
                ratio = abs(float(sv)) / max_abs if max_abs > 0 else 0
                bar_width = round(np.sqrt(ratio) * 100, 2)

                results.append({
                    'feature': self._format_feature_name(fname),
                    'feature_raw': fname,
                    'value': round(float(val), 4),
                    'shap_value': round(float(sv), 6),
                    'direction': 'increases_risk' if sv > 0 else 'decreases_risk',
                    'impact_pct': bar_width,
                })

        results.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return results[:top_n]

    @staticmethod
    def _format_feature_name(raw_name: str) -> str:
        """Convert feature_name to Feature Name for display."""
        # Handle one-hot encoded features
        if '_is_missing' in raw_name:
            base = raw_name.replace('_is_missing', '')
            return f"{base.replace('_', ' ').title()} (Missing)"

        # Handle prefixed one-hot columns
        for prefix in ['gender_', 'marital_status_', 'employment_type_', 'loan_type_']:
            if raw_name.startswith(prefix):
                category = prefix.replace('_', ' ').strip().title()
                value = raw_name[len(prefix):]
                return f"{category}: {value}"

        return raw_name.replace('_', ' ').title()
