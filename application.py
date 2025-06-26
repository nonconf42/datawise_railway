import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from app import app as application

@application.route('/', defaults={'path': ''})
@application.route('/<path:path>')
def serve_react_app(path):
    build_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
    if path and os.path.exists(os.path.join(build_folder, path)):
        from flask import send_from_directory
        return send_from_directory(build_folder, path)
    else:
        from flask import send_from_directory
        return send_from_directory(build_folder, 'index.html')

if __name__ == "__main__":
    # Railway provides PORT environment variable
    port = int(os.environ.get('PORT', 8000))
    application.run(host='0.0.0.0', port=port)