import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from PIL import Image
import numpy as np
import io

app = FastAPI(title="Face Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight model once at startup
model_name = "Facenet"
detector_backend = "opencv"
model = DeepFace.build_model(model_name)

def to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj

@app.get("/")
def root():
    return {"status": "Face API running"}

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save("temp.jpg")

        analysis = DeepFace.analyze(
            img_path="temp.jpg",
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
            detector_backend=detector_backend,
            models=model
        )[0]

        analysis = to_python(analysis)

        return {
            "age": analysis["age"],
            "gender": analysis["gender"],
            "emotion": analysis["dominant_emotion"]
        }

    except Exception as e:
        return {"error": str(e)}
