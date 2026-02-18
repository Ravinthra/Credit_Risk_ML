"""
Predictor app URL configuration.

WHY app-level urls?
- Each Django app owns its own routes
- Root urls.py includes this via include('predictor.urls')
- Makes the app self-contained and portable
"""
from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.predict_view, name='predict'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
]
