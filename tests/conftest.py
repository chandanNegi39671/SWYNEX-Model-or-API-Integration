import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
# Empty (not removed) so python-dotenv in app/llm.py cannot load a real key from .env during tests.
os.environ["GEMINI_API_KEY"] = ""