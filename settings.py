import os
from dotenv.main import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Server configs
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', 8000))
ALLOWED_MIME_TYPES = set(os.getenv('ALLOWED_MIME_TYPES', 'image/jpeg,image/png').split(','))

# Input parameters
INPUT_WIDTH = os.getenv('INPUT_WIDTH', 224)
INPUT_HEIGHT = os.getenv('INPUT_HEIGHT', 224)
INPUT_CHANNEL = os.getenv('INPUT_CHANNEL', 3)

# Model configs
MODEL_PATH = os.getenv('MODEL_PATH', 'model.onnx')