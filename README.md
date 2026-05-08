# Credit Risk ML — Loan Default Prediction System

> End-to-end credit scoring pipeline: SQL feature engineering → 4-model ensemble comparison → business-aware threshold optimization → Django + FastAPI deployment on Render.

🌐 **Live Demo:** [credit-risk-ai.onrender.com](https://credit-risk-ai.onrender.com)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.1-green?logo=django&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-purple?logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

---

## About This Project

Built to simulate how a real fintech company engineers, trains, and serves a credit scoring model. The system covers:

- **SQL-first feature engineering** from 4 normalized database tables (66 engineered features)
- **Production-grade preprocessing pipeline** with MNAR-aware imputation, ordinal + one-hot encoding, and dual scaling strategies
- **4-model comparison** with `RandomizedSearchCV` tuning and `StratifiedKFold` cross-validation
- **Business-aware threshold optimization** using a cost matrix calibrated to real lending economics (₹91.2L savings demonstrated)
- **Versioned model registry** with rollback support and metadata tracking
- **Django web app** with SHAP explainability, animated prediction gauge, and model dashboard
- **FastAPI ML microservice** with Pydantic validation, Swagger docs, and health monitoring
- **Full deployment stack:** Docker Compose (local) + Render Blueprint (cloud)

---

## Project Architecture

```
Credit_Risk_ML/
├── database/                     # Data generation & SQL feature engineering
│   ├── schema.sql                # 4-table normalized schema (customers, loans, credit_history, payments)
│   ├── feature_engineering.sql   # 66 features via SQL aggregations & window functions
│   ├── generate_data.py          # Initial synthetic dataset generator
│   ├── generate_realistic_data.py # Realistic data with noise, skew, and missing patterns
│   └── build_features.py         # Executes SQL → exports ml_features.csv
│
├── src/                          # Core ML pipeline
│   ├── preprocessing.py          # MissingValueHandler, FeatureEncoder, FeatureScaler, PreprocessingPipeline
│   ├── train_models.py           # ModelTrainer with RandomizedSearchCV for 4 models
│   ├── business_threshold.py     # CostMatrix, ThresholdOptimizer — cost-sensitive threshold selection
│   └── model_registry.py        # Versioned model storage with metadata & rollback
│
├── models/                       # Trained model artifacts (.pkl)
│   ├── Logistic_Regression_model.pkl
│   ├── XGBoost_model.pkl
│   ├── LightGBM_model.pkl
│   ├── Sklearn_GBM_model.pkl
│   ├── feature_scaler.pkl
│   └── registry/                 # Versioned registry entries
│
├── results/                      # Training outputs & reports
│   ├── model_comparison.csv      # AUC-ROC, F1, KS, Gini across all models
│   ├── training_report.json      # Full metrics + top-10 features per model
│   ├── threshold_analysis.png    # 3-panel threshold sweep visualization
│   ├── threshold_optimization.json
│   └── {Model}_classification_report.txt (×4)
│
├── ml_service/                   # FastAPI microservice
│   ├── main.py                   # App + /predict, /health, /model-info endpoints
│   ├── predictor.py              # MLPredictor with SHAP integration
│   ├── schemas.py                # Pydantic schemas (LoanApplication, PredictionResponse)
│   ├── requirements.txt
│   └── Dockerfile
│
├── data/                         # Raw & processed datasets
│   ├── customers.csv             # ~5K customer profiles
│   ├── loan_applications.csv     # ~10K loan records
│   ├── credit_history.csv        # Bureau features
│   ├── repayment_history.csv     # Monthly payment records
│   ├── ml_features.csv           # Final 66-feature dataset (SQL output)
│   └── processed/                # Train/test splits (X_train, X_test, scaled variants)
│
├── webapp/                       # Django web application
│   ├── config/                   # Settings, URLs, WSGI
│   └── predictor/                # Prediction Django app
│       ├── views.py              # Form handling & ML inference
│       ├── forms.py              # LoanApplicationForm with validation
│       ├── ml/                   # Model loader & SHAP explainer
│       ├── templates/            # predict.html, dashboard.html
│       └── static/               # CSS, JS (glassmorphism UI)
│
├── docker-compose.yml            # 3-service stack: Django + FastAPI + PostgreSQL
├── render.yaml                   # Render Blueprint (one-click cloud deploy)
├── build.sh                      # Render build script (migrations + collectstatic)
├── requirements.txt              # Production dependencies
└── .env.example                  # Environment variable template
```

---

## Model Performance

All four models were tuned with `RandomizedSearchCV` (5-fold stratified CV, scoring = AUC-ROC). Results on held-out 20% test set:

| Model | AUC-ROC | CV AUC | AUC-PR | Precision | Recall | F1 | KS Stat | Gini |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** 🏆 | **0.8294** | **0.8494** | 0.7065 | 0.6043 | 0.7197 | 0.6570 | 0.5165 | 0.6588 |
| XGBoost | 0.8293 | 0.8465 | 0.7035 | 0.6672 | 0.6417 | 0.6542 | 0.5149 | 0.6586 |
| LightGBM | 0.8275 | 0.8454 | 0.6996 | 0.6121 | 0.7086 | 0.6568 | 0.5101 | 0.6551 |
| Sklearn GBM | 0.8253 | 0.8462 | 0.6970 | 0.7067 | 0.5064 | 0.5900 | 0.5064 | 0.6506 |

> **KS Statistic > 0.40** is considered "good" in industry scorecards. All four models achieved KS ≈ 0.50–0.52. **Gini > 0.60** is "excellent" — all models landed at 0.65+.

### Why AUC ≈ 0.83 and not 0.99?

The initial synthetic dataset produced AUC ≈ 0.999 due to feature leakage — post-loan behavioral signals (missed payments, delinquencies) were included as input features. **13 leaky features were removed** and signal-to-noise ratio was tuned to achieve the realistic credit scorecard range (0.75–0.85). This is intentional: real-world banking models from the RBI/Basel II era typically score 0.70–0.85 AUC.

### Why Logistic Regression as Champion?

Logistic Regression achieved the **highest CV AUC (0.8494)** despite simpler architecture. More importantly, it is:
- **Interpretable by design** — coefficients show exact feature contribution direction and magnitude
- **Regulatory-friendly** — Basel II/III and RBI guidelines often require explainable models for credit decisions
- **Fast at inference** — microsecond predictions, critical for real-time loan approvals

The gradient boosting models were statistically equivalent but not significantly better, which indicates the signal is primarily in the features — not the model complexity.

---

## ML Engineering Deep Dive

### 1. SQL-Based Feature Engineering (`database/`)

Features are computed directly from 4 normalized tables using SQL aggregations, window functions, and self-joins. This mirrors production data warehouse workflows (dbt, BigQuery, Redshift).

**Key engineered features:**
| Feature | Source | Type |
|---|---|---|
| `dti_ratio` | EMI / annual income | Ratio |
| `emi_to_income_ratio` | Monthly EMI / monthly income | Ratio |
| `credit_utilization` | Outstanding / credit limit | Ratio (0–1) |
| `late_payment_ratio` | Late payments / total payments | Ratio |
| `employment_stability_ratio` | Years employed / age | Ratio |
| `income_per_dependent` | Income / (dependents + 1) | Normalized |
| `loan_to_income_ratio` | Loan amount / annual income | Ratio |
| `credit_score_tier` | Bucketed credit score | Ordinal |
| `is_thin_file` | Missing credit history flag | Binary |

Total: **66 features** from 4 tables (customers, loan_applications, credit_history, repayment_history).

---

### 2. MNAR-Aware Preprocessing Pipeline (`src/preprocessing.py`)

Three composable classes, each with `fit()` / `transform()` / `fit_transform()`:

**`MissingValueHandler`** — Handles Missing Not At Random (MNAR) patterns:
- A missing `credit_score` means the customer is new-to-credit — a risk signal in itself
- Creates binary `{col}_is_missing` indicator columns **before** imputation to preserve the missingness signal
- Numeric: imputed with median from training data only
- Categorical: imputed with mode from training data only

**`FeatureEncoder`** — Encoding strategy by feature type:
- **Ordinal**: `education`, `credit_score_tier`, `age_group`, `utilization_bucket`, `term_bucket`, `rate_tier` — preserves monotonic ordering that OHE would destroy
- **One-Hot**: `gender`, `marital_status`, `employment_type`, `loan_type` — nominal categories with no meaningful order
- Handles unseen categories at inference gracefully (fills 0, maps to −1)

**`FeatureScaler`** — Three scaling strategies:
- **StandardScaler** (z-score): normally-distributed features like `age`, `credit_score`, `loan_term_months`
- **RobustScaler** (median/IQR): outlier-prone features like `annual_income`, `loan_amount`, `emi_amount`
- **No scaling**: binary flags, bounded ratios (0–1), ordinal-encoded integers, OHE columns

**`PreprocessingPipeline`** orchestrates all steps in the correct order:
1. Separate target (`is_default`) and drop ID columns
2. Stratified train/test split (80/20, `random_state=42`) — **split happens before any fitting**
3. `fit_transform(X_train)` → `transform(X_test)` for each handler
4. Produces **two output versions**: scaled (for Logistic Regression) and unscaled (for tree models)

---

### 3. Model Training & Tuning (`src/train_models.py`)

**Why `RandomizedSearchCV` over `GridSearchCV`?**  
XGBoost's parameter space alone has ~92,000 combinations × 5 folds = 460,800 model fits. RandomizedSearchCV samples 50 random combinations (≈0.05% of the grid) and achieves near-optimal results per Bergstra & Bengio (2012). Production systems use Bayesian optimization (Optuna/Hyperopt) — 3–5× more efficient still.

**Model-specific tuning rationale:**

| Model | Key Parameters | Why |
|---|---|---|
| Logistic Regression | `C`, `penalty (L1/L2)` | L1 drives correlated features to zero (feature selection); L2 shrinks all coefficients |
| XGBoost | `max_depth`, `gamma`, `min_child_weight`, `scale_pos_weight` | `gamma` prunes splits; `scale_pos_weight` handles class imbalance |
| LightGBM | `num_leaves`, `min_child_samples`, histogram binning | `num_leaves` is the primary complexity control for leaf-wise growth |
| Sklearn GBM | `min_samples_leaf`, `min_samples_split`, `max_features` | Conservative depth-wise growth; good baseline for reproducibility |

**Metrics computed per model:**  
AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy, **KS Statistic**, **Gini Coefficient** — the last two are industry-standard credit scorecard metrics rarely seen in Kaggle notebooks.

---

### 4. Business Threshold Optimization (`src/business_threshold.py`)

The default 0.5 threshold assumes that missing a defaulter and rejecting a good customer cost the same. In lending, they emphatically do not.

**Cost Matrix:**
| Error Type | Description | Cost |
|---|---|---|
| False Negative (FN) | Approved a customer who defaulted | ₹1,00,000 |
| False Positive (FP) | Rejected a customer who would have repaid | ₹10,000 |
| True Negative / Positive | Correct decisions | ₹0 |

**10:1 asymmetry** → optimal threshold shifts left (more aggressive at catching defaults).

**Results on LightGBM test set:**

| | Default Threshold (0.50) | Optimal Threshold (0.11) |
|---|---|---|
| Total Business Cost | ₹2,11,20,000 | ₹1,20,00,000 |
| Recall (defaulters caught) | 70.86% | **98.57%** |
| Precision | 61.21% | 35.80% |
| Missed Defaulters (FN) | 183 | **9** |
| **Estimated Savings** | — | **₹91.2 Lakhs (43.2%)** |

The `ThresholdOptimizer` sweeps thresholds from 0.05 to 0.95, computes total business cost at each step, and generates a 3-panel analysis plot (cost curve, recall curve, precision curve).

---

### 5. Versioned Model Registry (`src/model_registry.py`)

Each trained model version is stored with:
- Serialized model + scaler (`.pkl` via `joblib`)
- Metadata: AUC-ROC, training timestamp, feature count, hyperparameters, optimal threshold
- Supports **rollback** to any previous version
- Singleton pattern for efficient loading in Django + FastAPI

---

### 6. FastAPI ML Microservice (`ml_service/`)

A stateless inference service designed to scale horizontally independently of the Django UI:

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Accepts `LoanApplication` JSON → returns default probability, risk tier, SHAP explanations, EMI estimate |
| `/health` | GET | Service health + model load status |
| `/model-info` | GET | Loaded model version, AUC, optimal threshold, available registry versions |
| `/docs` | GET | Auto-generated Swagger/OpenAPI UI |

**Request/response validated via Pydantic v2.** Startup uses FastAPI's `lifespan` context manager (model loaded at startup, graceful degradation if load fails).

---

### 7. Django Web Application (`webapp/`)

- **Predict page** — `LoanApplicationForm` with 15 fields; inline validation; animated gauge showing default probability
- **Dashboard** — Stripe/Linear-inspired model comparison table; top feature importance bars (sqrt-scaled for visual balance); SHAP explanation panel
- **Glassmorphism UI** with dark background, micro-animations, responsive layout
- **Static files** served via WhiteNoise (production-ready, no Nginx required)

---

## Quick Start — Local

```bash
# 1. Clone
git clone https://github.com/Ravinthra/Credit_Risk_ML.git
cd Credit_Risk_ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset (SQLite database + CSVs)
python database/generate_realistic_data.py

# 4. Build ML features via SQL
python database/build_features.py

# 5. Preprocess (creates data/processed/ splits)
python src/preprocessing.py

# 6. Train all 4 models (takes ~8 min on CPU)
python src/train_models.py

# 7. Optimize decision threshold
python src/business_threshold.py

# 8. Launch Django app
cd webapp
python manage.py migrate
python manage.py runserver
# → http://127.0.0.1:8000

# Optional: Run FastAPI service in parallel
cd ..
uvicorn ml_service.main:app --host 0.0.0.0 --port 8001 --reload
# → http://127.0.0.1:8001/docs
```

---

## Docker Deployment (Full Stack)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env: DJANGO_SECRET_KEY, POSTGRES_PASSWORD

# Build and start all 3 services
docker-compose up --build
```

**Services:**

| Service | Container | URL |
|---|---|---|
| Django Web UI | `creditrisk_django` | `http://localhost:8000` |
| FastAPI ML API | `creditrisk_ml_api` | `http://localhost:8001/docs` |
| PostgreSQL 16 | `creditrisk_db` | Internal (port 5432) |

The Django container mounts `./models` and `./results` as read-only volumes. The FastAPI container reads from the same model registry. PostgreSQL health-checks before Django starts.

---

## Cloud Deployment — Render

### Option A: Blueprint (Recommended)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Click **Apply** — live in ~3 minutes

### Option B: Manual Web Service

| Setting | Value |
|---|---|
| **Runtime** | Python 3.12 |
| **Build Command** | `./build.sh` |
| **Start Command** | `cd webapp && gunicorn config.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Environment Variables** | `DJANGO_SECRET_KEY` (generate), `DJANGO_DEBUG=False`, `DJANGO_ALLOWED_HOSTS=.onrender.com` |

The `build.sh` script installs dependencies, runs `collectstatic`, and runs database migrations automatically on each deploy.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | ✅ | Django secret key (generate with `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`) |
| `DJANGO_DEBUG` | ✅ | `False` in production |
| `DJANGO_ALLOWED_HOSTS` | ✅ | Comma-separated hostnames (e.g., `.onrender.com`) |
| `DATABASE_URL` | Docker only | PostgreSQL connection string |
| `ML_SERVICE_URL` | Docker only | FastAPI base URL (e.g., `http://ml_api:8001`) |
| `POSTGRES_DB` | Docker only | PostgreSQL database name |
| `POSTGRES_USER` | Docker only | PostgreSQL username |
| `POSTGRES_PASSWORD` | Docker only | PostgreSQL password |

Copy `.env.example` to `.env` and fill in values before running locally or in Docker.

---

## Interview Talking Points

**Why SQL instead of pandas for feature engineering?**
> In production, features live in data warehouses (BigQuery, Redshift, Snowflake). SQL pipelines are scalable, version-controllable via dbt, and reproducible across teams. Pandas-based pipelines don't scale past a single machine and are harder to schedule in Airflow/Prefect.

**Why not just use XGBoost and skip Logistic Regression?**
> Logistic Regression is a performance floor — if a boosting model barely beats it, the problem is with features, not architecture. Additionally, interpretable models are often legally required for credit decisions (Basel II, ECOA, RBI guidelines). LogReg with L1 also acts as embedded feature selection.

**How do you prevent training-serving skew?**
> The preprocessing pipeline uses `fit()` on training data only and `transform()` on test/production data. Scalers, encoders, and imputation values are serialized to disk via joblib and loaded at inference time. The FastAPI service loads the same scaler artifact that was fit during training — no recomputation.

**Why threshold = 0.11 and not 0.5?**
> Because costs are asymmetric: a missed defaulter on an unsecured personal loan costs ~₹1L in write-offs; rejecting a good customer costs ~₹10K in lost interest. With a 10:1 FN/FP cost ratio, the expected-cost-minimizing threshold shifts aggressively left. At 0.11, recall jumps from 70.9% to 98.6%, catching 174 additional defaulters at the cost of more false alarms — but the math clearly favors it.

**What metrics matter for credit risk?**
> AUC-ROC for threshold-independent discriminative ability, KS Statistic (max TPR−FPR separation) and Gini Coefficient (= 2×AUC−1) as industry-standard scorecard KPIs. KS > 0.40 is "good", Gini > 0.60 is "excellent" — all four models achieved both. Accuracy is largely meaningless at 31% class imbalance.

**Why not CatBoost as the 4th model?**
> CatBoost's ordered boosting would reduce target leakage in the boosting process and its native categorical handling would eliminate OHE entirely. It requires Python ≤ 3.12 for prebuilt wheels. Sklearn GBM serves as a conservative, fully reproducible sklearn-native baseline in its place. In a production system, I'd replace Sklearn GBM with CatBoost.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Web Framework | Django 5.1 + Gunicorn |
| ML API | FastAPI 0.109 + Uvicorn |
| ML Libraries | scikit-learn 1.3, XGBoost 2.0, LightGBM 4.0 |
| Explainability | SHAP 0.43 |
| Data | pandas 2.1, NumPy 1.26 |
| Database | SQLite (local ML) / PostgreSQL 16 (Docker/Render) |
| Serialization | joblib |
| Static Files | WhiteNoise |
| Containerization | Docker + Docker Compose |
| Cloud | Render (Blueprint) |

---

## About

**Ravinthra Amulraj**  
MCA Graduate · Python Developer · Aspiring ML Engineer

Built to demonstrate real-world ML engineering beyond Kaggle notebooks — SQL feature engineering, modular preprocessing, cost-sensitive optimization, microservice architecture, and cloud deployment working together as a coherent system.

- **GitHub:** [github.com/Ravinthra](https://github.com/Ravinthra)
- **Live Project:** [credit-risk-ai.onrender.com](https://credit-risk-ai.onrender.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.