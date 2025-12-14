import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import io
from PIL import Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_python(obj):
    """Recursively converts numpy types → Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj

@app.get("/")
async def root():
    return {"message": "Face API running"}

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.save("temp.jpg")

        # -------- DeepFace ANALYZE --------
        analysis_raw = DeepFace.analyze(
            img_path="temp.jpg",
            actions=["age", "gender", "emotion"],
            enforce_detection=True
        )[0]

        # Convert analysis to Python-safe JSON
        analysis = to_python(analysis_raw)

        # -------- DeepFace EMBEDDING --------
        emb_raw = DeepFace.represent(
            img_path="temp.jpg",
            model_name="Facenet"
        )[0]["embedding"]

        embedding = to_python(emb_raw)

        return {
            "age": analysis["age"],
            "gender": analysis["gender"],
            "emotion": analysis["dominant_emotion"],

        }

    except Exception as e:
        return {"error": str(e)}
