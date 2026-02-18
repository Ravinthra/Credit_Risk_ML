#!/usr/bin/env python
"""
Django management script for Credit Risk Prediction webapp.

WHY manage.py?
- Entry point for all Django CLI commands (runserver, migrate, etc.)
- Points to our config.settings module
"""
import os
import sys


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure it's installed and "
            "available on your PYTHONPATH environment variable."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
