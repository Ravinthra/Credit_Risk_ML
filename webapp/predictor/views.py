"""
Credit Risk ML -- Django Views
===============================
Thin views that delegate ML logic to the ml/ service layer.

WHY THIN VIEWS?
- Views handle HTTP concerns (request/response, rendering)
- ML logic lives in ml/ (model loading, preprocessing, explanation)
- This separation means:
  1. Views can be tested without loading the ML model
  2. ML logic can be tested without Django
  3. Swapping Django for FastAPI only changes this file

Interview Insight:
"How do you structure ML-powered web apps?"
-> "I follow the Service Layer pattern. Views are thin controllers
   that validate input and call service functions. The ML pipeline
   (loading, preprocessing, prediction, explanation) lives in a
   separate module. This makes the codebase testable, maintainable,
   and framework-agnostic."
"""

import json
from django.shortcuts import render
from django.conf import settings

from .forms import LoanApplicationForm
from .ml.model_loader import ModelLoader
from .ml.preprocessor import InferencePreprocessor
from .ml.explainer import ShapExplainer


# WHY instantiate here instead of in each view?
# These are stateless (preprocessor, explainer) or singleton (loader).
# Creating them once at module level avoids per-request overhead.
preprocessor = InferencePreprocessor()
explainer = ShapExplainer()


def predict_view(request):
    """
    Main prediction view.

    GET: Shows empty loan application form
    POST: Processes form, runs ML prediction, returns results

    WHY function-based view instead of class-based?
    - For simple form handling, FBV is more readable
    - CBV adds overhead for a single-page form
    - In production with complex CRUD, use CBV (CreateView, etc.)
    """
    result = None
    explanations = None

    if request.method == 'POST':
        form = LoanApplicationForm(request.POST)

        if form.is_valid():
            # 1. Get ML model
            loader = ModelLoader.get_instance()

            # 2. Preprocess form data -> 66-feature vector
            features_df = preprocessor.preprocess(form.cleaned_data)

            # 3. Get prediction
            probability = loader.predict_proba(features_df)
            risk_info = loader.get_risk_category(probability)

            # 4. Get SHAP explanation
            model = loader.get_model()
            explanations = explainer.explain(model, features_df, top_n=10)

            # 5. Compute EMI for display
            emi = InferencePreprocessor._compute_emi(
                form.cleaned_data['loan_amount'],
                form.cleaned_data['interest_rate'],
                form.cleaned_data['loan_term_months'],
            )

            result = {
                'probability': round(probability * 100, 2),
                'probability_raw': probability,
                'category': risk_info['category'],
                'color': risk_info['color'],
                'description': risk_info['description'],
                'action': risk_info['action'],
                'emi': round(emi, 2),
                'model_used': settings.ML_CONFIG['default_model'],
            }
    else:
        form = LoanApplicationForm()

    context = {
        'form': form,
        'result': result,
        'explanations': explanations,
        'explanations_json': json.dumps(explanations) if explanations else '[]',
    }

    return render(request, 'predictor/predict.html', context)


def dashboard_view(request):
    """
    Model performance dashboard.

    Shows comparison metrics for all 4 trained models,
    training report details, and feature importance.
    """
    loader = ModelLoader.get_instance()

    comparison = loader.get_model_comparison()
    report = loader.get_training_report()

    # Get feature importance for the best model
    best_model = report.get('best_model', 'LightGBM')
    model_details = report.get('model_details', {})
    best_details = model_details.get(best_model, {})
    top_features = best_details.get('top_10_features', [])

    context = {
        'comparison': comparison,
        'report': report,
        'best_model': best_model,
        'best_auc': report.get('best_auc_roc', 0),
        'top_features': top_features,
        'model_details': model_details,
        'comparison_json': json.dumps(comparison),
        'features_json': json.dumps(top_features),
    }

    return render(request, 'predictor/dashboard.html', context)
