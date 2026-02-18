# ── FastAPI ML Service ──
FROM python:3.12-slim AS ml-api

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY ml_service/requirements.txt ./ml_service/requirements.txt
RUN pip install --no-cache-dir -r ml_service/requirements.txt

# Copy project
COPY ml_service/ ./ml_service/
COPY models/ ./models/
COPY results/ ./results/
COPY src/ ./src/
COPY webapp/predictor/ml/ ./webapp/predictor/ml/

# Needed for Django imports in predictor.py
COPY webapp/predictor/__init__.py ./webapp/predictor/__init__.py
COPY webapp/predictor/templatetags/ ./webapp/predictor/templatetags/
COPY webapp/config/ ./webapp/config/

WORKDIR /app

EXPOSE 8001

CMD ["uvicorn", "ml_service.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "2"]
