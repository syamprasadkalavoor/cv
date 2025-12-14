import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf

# -----------------------------
# TensorFlow optimizations
# -----------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Face Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Facenet model from local folder
# -----------------------------
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
MODEL_DIR = "./deepface_models"  # local folder in repo
facenet_model = DeepFace.build_model(MODEL_NAME, model_dir=MODEL_DIR)

# -----------------------------
# ThreadPool for async-safe blocking calls
# -----------------------------
executor = ThreadPoolExecutor(max_workers=1)

# -----------------------------
# Helper function to convert numpy types to Python types
# -----------------------------
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

# -----------------------------
# Async wrapper for DeepFace
# -----------------------------
async def analyze_face_thread(image_array):
    loop = asyncio.get_event_loop()

    # Analyze age, gender, emotion
    analysis = await loop.run_in_executor(
        executor,
        lambda: DeepFace.analyze(
            img_path=image_array,
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )[0]
    )

    # Get face embedding
    embedding = await loop.run_in_executor(
        executor,
        lambda: DeepFace.represent(
            img_path=image_array,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )[0]["embedding"]
    )

    return to_python(analysis), to_python(embedding)

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"status": "Face API running"}

# -----------------------------
# Analyze face endpoint
# -----------------------------
@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Analyze face
        analysis, embedding = await analyze_face_thread(image_array)

        return {
            "age": analysis["age"],
            "gender": analysis["gender"],
            "emotion": analysis["dominant_emotion"],
            "embedding": embedding
        }

    except Exception as e:
        return {"error": str(e)}
