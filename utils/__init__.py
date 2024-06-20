import logging
import os
import time
from typing import Callable
import onnxruntime

def retry(func, num_retries=3, delay=1) -> Callable:
    def wrapper(*args, **kwargs):
        for i in range(num_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.info(f"Attempt {i + 1} failed: {e}")
                time.sleep(delay)

        raise Exception(f"Failed to execute {func.__name__} after {num_retries} attempts")
    
    return wrapper

# Load the ONNX model
@retry
def load_model(model_path: str):
    if not os.path.exists(model_path):
        logging.info(f"Model file not found: {model_path}")

    return onnxruntime.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if onnxruntime.get_device() == 'GPU' else ["CPUExecutionProvider"]
    )