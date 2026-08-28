"""
wsgi.py
=======
WSGI entrypoint for production deployments (Render, Railway, Heroku, Gunicorn).
"""

from app.server import app

if __name__ == "__main__":
    app.run()
