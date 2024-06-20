import uvicorn
from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import os
import io
import time
import mimetypes
import logging
import base64
from PIL import Image

from utils.yolov8_seg import YOLOv8Seg

from settings import HOST, PORT, ALLOWED_MIME_TYPES, MODEL_PATH

logging.basicConfig(level=logging.INFO)
app = FastAPI()
model = YOLOv8Seg(MODEL_PATH)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the response model

class InstanceSegmentationResponse(BaseModel):
    segments: list[list[tuple[int, int]]]

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

    contents = [await image.read() for image in images]
    images = np.stack([np.array(Image.open(io.BytesIO(content))) for content in contents])
    print(images.shape)

    # Convert the image to numpy array
    start_time = time.time()
    prediction = model(images[0])
    end_time = time.time()

    print(prediction)

    # Calculate inference time
    inference_time = end_time - start_time

    # Return the prediction result and inference time
    return {
        "predictions": prediction,
        "inference_time": inference_time
    }

@app.post("/predict")
async def post_predict(images: list[UploadFile]):
    # Read the image file
    for image in images:
        if image.content_type not in ALLOWED_MIME_TYPES:
            return Response(status_code=415, content="Unsupported Media Type")
    
    contents = [await image.read() for image in images]

    # Convert the image to numpy array
    image_array = np.frombuffer(contents, dtype=np.uint8)

    # Perform prediction using the onnxruntime model
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    start_time = time.time()
    prediction = model.run([output_name], {input_name: image_array})
    end_time = time.time()

    print(prediction[2].shape)

    # save masks to disk
    for i, mask in enumerate(prediction[2]):
        mask = mask.squeeze()
        mask = Image.fromarray(mask)
        mask.save(f"mask_{i}.png")

    # Calculate inference time
    inference_time = end_time - start_time

    # Return the prediction result and inference time
    return {
        "prediction": prediction,
        "inference_time": inference_time
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
