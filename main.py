import uvicorn, os, io, time, mimetypes, logging, base64
from fastapi import FastAPI, UploadFile, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from uuid import uuid4
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from settings import HOST, PORT, UPLOADS_PATH, RESULTS_PATH, ALLOWED_MIME_TYPES, MODEL_PATH

logging.basicConfig(level=logging.INFO)
app = FastAPI()
model = YOLO(MODEL_PATH)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InstanceSegmentationResponse(BaseModel):
    segments: list[list[tuple[int, int]]]

# Define the API endpoint
@app.get("/health")
async def get_health():
    if model is None:
        return JSONResponse(status_code=500, content="Model not loaded")

    return JSONResponse(status_code=200, content="Model loaded successfully")

@app.post('/predict')
async def post_predict(image: UploadFile):
    # Read the image file
    guessed_type, _ = mimetypes.guess_type(image.filename)
    if image.content_type not in ALLOWED_MIME_TYPES or guessed_type not in ALLOWED_MIME_TYPES:
        return Response(status_code=415, content="Unsupported Media Type")

    content = await image.read()
    image = np.array(Image.open(io.BytesIO(content)))

    # Convert the image to numpy array
    start_time = time.time()
    prediction = model(image)
    end_time = time.time()

    # Calculate inference time
    inference_time = end_time - start_time

    pred = prediction[0]
    im = pred.plot()
    plt.axis('off')
    plt.imshow(im)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Return the prediction image
    #return Response(content=base64.encodebytes(im.tobytes()), media_type='image/png')
    return StreamingResponse(buf, media_type='image/png')

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
