"""
Vercel entrypoint for Flask application
This file is required by Vercel to find the Flask app
"""

from web_chat import app

# Vercel expects the app to be available
__all__ = ['app']
