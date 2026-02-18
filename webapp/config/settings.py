"""
Django Settings for Credit Risk Prediction Web Application.

WHY this structure?
- config/ package keeps settings separate from app code
- ML_MODELS_DIR points to the existing trained models
- Single-app architecture (predictor) keeps things simple
"""
import os
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # webapp/
PROJECT_ROOT = BASE_DIR.parent  # Credit_Risk_ML/

# WHY PROJECT_ROOT?
# Our trained models and data live one level above webapp/.
# This lets the Django app access them without copying files.
ML_MODELS_DIR = PROJECT_ROOT / 'models'
ML_RESULTS_DIR = PROJECT_ROOT / 'results'
ML_DATA_DIR = PROJECT_ROOT / 'data'

# ─── Security ───────────────────────────────────────────
# Read secrets from environment — never commit real keys to git
SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-dev-only-change-in-production'
)
DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() in ('true', '1', 'yes')
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', '*').split(',')

# ─── Application Definition ─────────────────────────────
# WHY only one app?
# Single Responsibility: 'predictor' handles all ML prediction logic.
# In production, you might add 'accounts', 'api', etc.
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'predictor',  # Our credit risk prediction app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Serve static files in production
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

# ─── Templates ──────────────────────────────────────────
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # App-level templates/ dirs are auto-discovered
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# ─── Database ───────────────────────────────────────────
# WHY SQLite?
# We don't need a database for predictions — models are loaded
# from pickle files. SQLite is just for Django's admin/sessions.
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ─── Static Files ───────────────────────────────────────
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# WHY WhiteNoise?
# Render (and most PaaS) don't serve static files via nginx.
# WhiteNoise serves them directly from Django with compression + caching.
STORAGES = {
    'staticfiles': {
        'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage',
    },
}

# ─── Default field type ─────────────────────────────────
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ─── ML Configuration ───────────────────────────────────
# WHY here and not in the ML module?
# Settings.py is the single source of truth for configuration.
# The ML module reads these to know which model to load.
ML_CONFIG = {
    'default_model': 'Logistic_Regression',  # Best performing (AUC: 0.8494)
    'available_models': {
        'LightGBM': ML_MODELS_DIR / 'LightGBM_model.pkl',
        'XGBoost': ML_MODELS_DIR / 'XGBoost_model.pkl',
        'Logistic_Regression': ML_MODELS_DIR / 'Logistic_Regression_model.pkl',
        'Sklearn_GBM': ML_MODELS_DIR / 'Sklearn_GBM_model.pkl',
    },
    'training_report': ML_RESULTS_DIR / 'training_report.json',
    'model_comparison': ML_RESULTS_DIR / 'model_comparison.csv',
}
