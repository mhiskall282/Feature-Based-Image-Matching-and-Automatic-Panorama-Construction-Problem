"""
api/index.py
============
Vercel serverless entrypoint.
Imports and exposes the Flask WSGI application instance.
"""

import sys
from pathlib import Path

# Add project root to Python search path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.server import app

# Vercel requires the WSGI app instance named 'app'
if __name__ == "__main__":
    app.run()
