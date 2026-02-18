"""
Predictor app configuration.

WHY AppConfig?
- Django uses this to auto-discover the app
- The 'name' must match the Python import path
"""
from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'
    verbose_name = 'Credit Risk Predictor'
