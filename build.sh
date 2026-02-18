#!/usr/bin/env bash
# Render Build Script
# This script runs during deployment to install deps and collect static files

set -o errexit  # Exit on error

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Collecting static files..."
cd webapp
python manage.py collectstatic --noinput

echo "==> Build complete!"
