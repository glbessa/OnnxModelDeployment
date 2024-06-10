import os
import dotenv

dotenv.load_dotenv()

# Server configs
HOST = os.getenv('HOST', '0.0.0.0')
PORT = os.getenv('PORT', 8000)

# Model configs
MODEL_PATH = os.getenv('MODEL_PATH', 'model.onnx')