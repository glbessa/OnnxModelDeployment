import uvicorn
from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime
import numpy as np
import os
import time
import mimetypes

from settings import HOST, PORT, ALLOWED_MIME_TYPES, MODEL_PATH

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 

# Load the ONNX model
model = None
if os.path.exists(MODEL_PATH):
    model = onnxruntime.InferenceSession(MODEL_PATH)

# Define the API endpoint
@app.get("/health")
async def get_health():
    if model is None:
        return Response(status_code=500, content="Model not loaded")

    return Response(status_code=200, content="Model loaded successfully")

@app.post('/predict')
async def post_predict(images: list[UploadFile]):
    # Read the image file
    for image in images:
        guessed_type, _ = mimetypes.guess_type(image.filename)
        if image.content_type not in ALLOWED_MIME_TYPES or guessed_type not in ALLOWED_MIME_TYPES:
            return Response(status_code=415, content="Unsupported Media Type")
    
    contents = await image.read()

    

    # Convert the image to numpy array
    start_time = time.time()
    prediction = model.run([output_name], {input_name: image_array})
    end_time = time.time()

    # Calculate inference time
    inference_time = end_time - start_time

    # Return the prediction result and inference time
    return {
        "prediction": prediction,
        "inference_time": inference_time
    }

@app.post("/predict")
async def post_predict(images: list[UploadFile]):
    # Read the image file
    for image in images:
        if image.content_type not in ALLOWED_MIME_TYPES:
            return Response(status_code=415, content="Unsupported Media Type")
    
    contents = await image.read()

    # Convert the image to numpy array
    image_array = np.frombuffer(contents, dtype=np.uint8)

    # Perform prediction using the onnxruntime model
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    start_time = time.time()
    prediction = model.run([output_name], {input_name: image_array})
    end_time = time.time()

    # Calculate inference time
    inference_time = end_time - start_time

    # Return the prediction result and inference time
    return {
        "prediction": prediction,
        "inference_time": inference_time
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
