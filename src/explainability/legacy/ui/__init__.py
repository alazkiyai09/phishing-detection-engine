"""
User interfaces for explanation display.

Includes Streamlit apps for end-users and security analysts.
"""

from src.explainability.legacy.ui.user_app import main as user_app_main
from src.explainability.legacy.ui.analyst_interface import main as analyst_interface_main

__all__ = [
    "user_app_main",
    "analyst_interface_main",
]
