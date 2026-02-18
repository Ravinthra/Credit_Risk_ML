# 🚀 Credit Risk / Loan Default Prediction System

End-to-end Credit Risk ML system built to simulate real-world fintech engineering workflows — from SQL-based feature engineering to model deployment using Django & FastAPI.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.1-green?logo=django&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

---

## 👨‍💻 About This Project

This project was built after completing a Customer Churn Prediction system to deepen my understanding of:

- Financial risk modeling
- SQL-based feature engineering
- Imbalanced classification
- Business-aware threshold optimization
- ML model deployment with Django & FastAPI
- Cloud deployment with Render

The goal was to simulate how a real fintech company designs, trains, and serves a credit scoring model.

---

## 🏗️ Project Architecture

```
Credit_Risk_ML/
├── database/                  # Data generation & SQL feature engineering
│   ├── schema.sql
│   ├── feature_engineering.sql
│   ├── generate_data.py
│   ├── generate_realistic_data.py
│   └── build_features.py
├── src/                       # ML pipeline
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── business_threshold.py
│   └── model_registry.py
├── models/                    # Trained model .pkl files
├── results/                   # Training reports & comparisons
├── ml_service/                # FastAPI microservice
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── webapp/                    # Django web application
│   ├── config/                # Django settings & URLs
│   └── predictor/             # Prediction app (views, forms, templates, ML)
│       ├── templates/         # HTML templates (predict, dashboard)
│       ├── static/            # CSS, JS
│       └── ml/                # Model loader & SHAP explainer
├── docker-compose.yml
├── render.yaml                # Render deployment blueprint
├── build.sh                   # Render build script
├── requirements.txt           # Production Python dependencies
├── .env.example               # Environment variable template
└── README.md
```

---

## 🎯 How This Improves Over My Customer Churn Project

| Aspect | Customer Churn | Credit Risk (This Project) |
|--------|---------------|---------------------------|
| **Domain** | Telecom / CRM | FinTech / Banking |
| **Feature Engineering** | Pandas-based | SQL-first (66 engineered features) |
| **Data Complexity** | Single table | 4 normalized tables |
| **Class Imbalance** | Moderate | ~31% default rate |
| **Business Impact** | Revenue retention | Direct financial loss (₹91.2L savings) |
| **Deployment** | Django ML integration | Django + FastAPI + Docker + Render |
| **Explainability** | — | SHAP with sqrt-scaled visualization |

---

## 📊 Model Performance

| Model | AUC-ROC | CV AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| **Logistic Regression** 🏆 | **0.8294** | **0.8494** | 0.6043 | 0.7197 | 0.6570 |
| XGBoost | 0.8293 | 0.8465 | 0.6672 | 0.6417 | 0.6542 |
| LightGBM | 0.8275 | 0.8454 | 0.6121 | 0.7086 | 0.6568 |
| Sklearn GBM | 0.8253 | 0.8462 | 0.7067 | 0.5064 | 0.5900 |

### Why AUC ≈ 0.83–0.85?

Initially, the synthetic dataset produced AUC ≈ 0.999 due to feature leakage (post-loan behavioral signals). I removed 13 leaky features and tuned signal-to-noise ratio to achieve a realistic credit scorecard performance range (0.75–0.85). This reflects real-world banking ML behavior.

---

## 🧠 Key ML Engineering Components

### 1️⃣ SQL-Based Feature Engineering

- Aggregations from multiple tables (customers, loans, credit_history, payments)
- Debt-to-Income (DTI), Credit Utilization, Payment delay ratios
- Missingness indicators (MNAR-aware)
- 66 features engineered from 4 normalized tables

### 2️⃣ MNAR-Aware Preprocessing Pipeline

- Missing value imputation with missingness indicators
- Encoding: One-hot + Ordinal strategies
- Scaling: StandardScaler / RobustScaler
- Strict train/test separation to avoid data leakage

### 3️⃣ Business-Aware Threshold Optimization

Instead of default 0.5 threshold:

- **False Negative** (Missed Defaulter) = ₹100,000 loss
- **False Positive** (Rejected Good Customer) = ₹10,000 loss
- **Optimal Threshold** = 0.11 → **₹91.2L estimated savings**

### 4️⃣ Versioned Model Registry

- Model + scaler + encoders stored per version
- Metadata: AUC, training date, feature count, hyperparameters
- Supports rollback and safe loading

### 5️⃣ Django Web Application

- Premium dark UI with glassmorphism & micro-animations
- Loan prediction form with animated gauge
- SHAP explainability bars (sqrt-scaled for visibility)
- Model comparison dashboard (Stripe/Linear-inspired design)
- Responsive design (mobile-friendly)

### 6️⃣ FastAPI ML Microservice

- `/predict` endpoint with Pydantic validation
- `/health` endpoint for monitoring
- Swagger documentation at `/docs`
- Production-ready with gunicorn

---

## 🚀 Quick Start (Local)

```bash
# Clone the repository
git clone https://github.com/Ravinthra/Credit_Risk_ML.git
cd Credit_Risk_ML

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python database/generate_realistic_data.py

# Build ML features
python database/build_features.py

# Preprocess & train
python src/preprocessing.py
python src/train_models.py

# Optimize threshold
python src/business_threshold.py

# Run Django app
cd webapp
python manage.py runserver
# Visit http://127.0.0.1:8000
```

---

## 🌐 Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and configures everything
5. Click **Apply** — your app goes live!

**Manual setup** (if not using Blueprint):

| Setting | Value |
|---------|-------|
| Build Command | `./build.sh` |
| Start Command | `cd webapp && gunicorn config.wsgi:application --bind 0.0.0.0:$PORT` |
| Environment | `DJANGO_SECRET_KEY` (auto-generate), `DJANGO_DEBUG=False` |

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

**Services:**

| Service | URL |
|---------|-----|
| Django UI | `localhost:8000` |
| FastAPI API | `localhost:8001/docs` |
| PostgreSQL | Internal |

---

## 🧪 What This Project Demonstrates

- Strong Python + SQL integration
- Imbalanced classification handling
- Feature leakage detection & prevention
- Business-aware ML evaluation (cost-sensitive thresholds)
- SHAP model explainability
- Full-stack deployment: Django + FastAPI + Docker + Render
- Production-ready security (env-based secrets, whitenoise static serving)

---

## 🎤 Interview-Ready Talking Points

**Why SQL instead of only pandas?**

> In production, feature engineering happens in databases or warehouses. SQL-based pipelines are scalable and reproducible.

**Why not high AUC?**

> Very high AUC usually indicates leakage. Realistic financial models typically range 0.70–0.85.

**How do you avoid training-serving skew?**

> Training uses fit_transform; inference uses transform-only with serialized preprocessing objects.

**Why Logistic Regression as champion?**

> Despite simpler architecture, it had the best CV AUC (0.8494) and is fully interpretable — critical for financial regulation compliance.

---

## 👨‍🎓 About Me

**Ravinthra Amulraj**
MCA Graduate | Python Developer | Aspiring ML Engineer

Skilled in:

- Python, SQL, Django, FastAPI
- Machine Learning & Financial Risk Modeling
- Docker & Cloud Deployment
- Data Engineering & Feature Engineering

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.