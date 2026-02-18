# ── Django Web Application ──
FROM python:3.12-slim AS django

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY webapp/requirements.txt ./webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt gunicorn psycopg2-binary

# Copy project
COPY webapp/ ./webapp/
COPY models/ ./models/
COPY results/ ./results/
COPY src/ ./src/

# Static files
WORKDIR /app/webapp
RUN python manage.py collectstatic --noinput 2>/dev/null || true

EXPOSE 8000

CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--timeout", "120"]
