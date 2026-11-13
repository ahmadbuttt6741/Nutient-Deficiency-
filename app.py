"""
AgriVision AI - Root Entry Point for Streamlit Community Cloud & Local Deployment.
Redirects to the production application in app/app.py.
"""
import sys
from pathlib import Path

# Ensure root directory is at the front of sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.app import main

if __name__ == "__main__":
    main()
