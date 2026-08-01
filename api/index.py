import os
import sys

# Get the directory of index.py (api/) and add its parent (project root) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Import the FastAPI app instance
from app.backend.main import app
