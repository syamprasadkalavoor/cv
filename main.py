<<<<<<< HEAD
import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import asyncio
from concurrent.futures import ThreadPoolExecutor

# TensorFlow optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# FastAPI setup
app = FastAPI(title="Face Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload Facenet model once
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
facenet_model = DeepFace.build_model(MODEL_NAME)

# ThreadPool for async-safe blocking calls
executor = ThreadPoolExecutor(max_workers=1)

# Helper to convert numpy types to Python types
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

# Async wrapper to run DeepFace blocking calls
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

    # Get face embedding vector
    embedding = await loop.run_in_executor(
        executor,
        lambda: DeepFace.represent(
            img_path=image_array,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )[0]["embedding"]
    )

    analysis = to_python(analysis)
    embedding = to_python(embedding)
    return analysis, embedding

# Root endpoint
@app.get("/")
def root():
    return {"status": "Face API running"}

# Analyze face endpoint
@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Read image from upload
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Run DeepFace analysis async
        analysis, embedding = await analyze_face_thread(image_array)

        return {
            "age": analysis["age"],
            "gender": analysis["gender"],
            "emotion": analysis["dominant_emotion"],
            "embedding": embedding
        }

    except Exception as e:
        return {"error": str(e)}
=======
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
          
        )[0]

        analysis = to_python(analysis)

        return {
            "age": analysis["age"],
            "gender": analysis["gender"],
            "emotion": analysis["dominant_emotion"]
        }

    except Exception as e:
        return {"error": str(e)}
>>>>>>> 7597e87cea79753217e1414ef445050dfb860af7
