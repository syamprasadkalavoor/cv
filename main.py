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
from typing import Dict, Any, List

# -----------------------------
# TensorFlow Optimizations
# -----------------------------
# Suppress oneDNN messages and set log level to error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")
# Constrain thread usage for better performance in concurrent FastAPI/Uvicorn
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# -----------------------------
# Global Variables
# -----------------------------
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
MODELS: Dict[str, Any] = {}  # Dictionary to hold pre-loaded models

# Using max_workers=1 to dedicate a single thread for DeepFace/TensorFlow operations
executor = ThreadPoolExecutor(max_workers=1)

# -----------------------------
# FastAPI App Initialization
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
# App Startup and Shutdown Events
# -----------------------------

@app.on_event("startup")
def load_models_at_startup():
    """Builds and caches the core Facenet model when the application starts."""
    global MODELS
    print("Attempting to load DeepFace models...")

    try:
        # FIX: ONLY build the main embedding model (Facenet).
        # DeepFace.analyze() will handle the loading/caching of Age/Gender/Emotion models.
        MODELS["facenet"] = DeepFace.build_model(MODEL_NAME)

        # Optional: Forcing the cache of attribute models (without storing them globally)
        # by calling analyze on a dummy image can ensure faster first-request speed,
        # but is less critical than fixing the ValueError.

        print("DeepFace models loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load DeepFace models. Reason: {e}")
        # Re-raise the exception to prevent the application from starting in a broken state
        raise e


@app.on_event("shutdown")
def shutdown_executor():
    """Shuts down the ThreadPoolExecutor when the app shuts down."""
    print("Shutting down ThreadPoolExecutor...")
    executor.shutdown(wait=False)


# -----------------------------
# Helper Functions
# -----------------------------

def to_python(obj: Any) -> Any:
    """Recursively converts NumPy types (like int, float, array) to standard Python types for JSON serialization."""
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
# Core DeepFace Worker
# -----------------------------

async def analyze_face_thread(image_array: np.ndarray, models_dict: Dict[str, Any]) -> tuple[
    Dict[str, Any], List[float]]:
    """Executes DeepFace calls in a separate thread to prevent blocking the event loop."""
    loop = asyncio.get_event_loop()

    # DeepFace.analyze uses the models cached by the initial build_model call implicitly.
    analysis = await loop.run_in_executor(
        executor,
        lambda: DeepFace.analyze(
            img_path=image_array,
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )[0]
    )

    # DeepFace.represent is explicitly passed the pre-loaded Facenet model.
    embedding = await loop.run_in_executor(
        executor,
        lambda: DeepFace.represent(
            img_path=image_array,
            model=models_dict["facenet"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )[0]["embedding"]
    )

    return to_python(analysis), to_python(embedding)


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "Face API running"}


@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """Accepts an image file and returns age, gender, emotion, and Facenet embedding."""
    try:
        # Read image bytes and open using PIL
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Offload the blocking analysis to the thread pool
        analysis, embedding = await analyze_face_thread(image_array, MODELS)

        return {
            "age": analysis.get("age"),
            "gender": analysis.get("gender"),
            "emotion": analysis.get("dominant_emotion"),
            "embedding": embedding,
            "message": "Face analysis successful"
        }

    except Exception as e:
        # Catch and handle errors during the request (e.g., no face detected)
        error_message = f"Face analysis failed. Reason: {str(e)}"
        print(f"Request failed: {error_message}")
        return {"error": error_message}