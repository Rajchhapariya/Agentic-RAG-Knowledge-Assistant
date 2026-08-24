"""
Streamlit Application Entrypoint for Agentic RAG Knowledge Assistant.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.app import run_app

run_app()
