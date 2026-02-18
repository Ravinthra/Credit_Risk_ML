"""
Root URL configuration.

WHY delegate to predictor.urls?
- Keeps URL definitions close to the app that handles them
- Root urls.py just orchestrates — the app owns its routes
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor.urls')),
]
