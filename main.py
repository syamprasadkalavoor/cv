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
# Constrain thread usage for better performance in concurrent FastAPI/Uvicorn
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
# Global Model Loading (Occurs once at startup)
# -----------------------------
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

# Build all required models explicitly. This avoids re-loading them on every request.
# NOTE: The 'model_dir' argument is removed to fix the deployment TypeError.
try:
    FACENET_MODEL = DeepFace.build_model(MODEL_NAME)
    AGE_MODEL = DeepFace.build_model("Age")
    GENDER_MODEL = DeepFace.build_model("Gender")
    EMOTION_MODEL = DeepFace.build_model("Emotion")
except Exception as e:
    print(f"Error loading DeepFace models: {e}")
    # You might want to raise an exception here to halt startup if models are critical

# -----------------------------
# ThreadPool for async-safe blocking calls
# -----------------------------
# Using max_workers=1 to ensure DeepFace/TensorFlow runs on a single thread
# (matching the tf config) and avoids global state issues.
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
async def analyze_face_thread(image_array, facenet_model, age_model, gender_model, emotion_model):
    """Executes DeepFace calls in a separate thread to prevent blocking the event loop."""
    loop = asyncio.get_event_loop()

    # Pass the pre-loaded models to DeepFace.analyze
    analysis = await loop.run_in_executor(
        executor,
        lambda: DeepFace.analyze(
            img_path=image_array,
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
            models={
                "age": age_model,
                "gender": gender_model,
                "emotion": emotion_model
            }
        )[0]
    )

    # Pass the pre-loaded model to DeepFace.represent
    embedding = await loop.run_in_executor(
        executor,
        lambda: DeepFace.represent(
            img_path=image_array,
            model=facenet_model,
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
        # Read image from file upload
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Convert PIL Image to NumPy array (preferred input for DeepFace)
        image_array = np.array(image)

        # Offload the blocking DeepFace analysis to the thread pool
        analysis, embedding = await analyze_face_thread(
            image_array,
            FACENET_MODEL,
            AGE_MODEL,
            GENDER_MODEL,
            EMOTION_MODEL
        )

        return {
            "age": analysis["age"],
            "gender": analysis["gender"],
            "emotion": analysis["dominant_emotion"],
            "embedding": embedding,
            "message": "Face analysis successful"
        }

    except Exception as e:
        # Log the detailed error, but return a simple message
        print(f"Analysis failed: {e}")
        return {"error": f"Face analysis failed. Reason: {str(e)}"}