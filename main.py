import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import numpy as np
import requests
from tensorflow.keras.models import load_model

from preprocess import (
    load_frames_from_video,
    decode_base64_image,
    build_clips,
)

MODEL_PATH = "keras_model.h5"

app = FastAPI(title="Violence Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow Lovable AI Frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None


@app.on_event("startup")
def load_model_at_startup():
    global model
    if not os.path.exists(MODEL_PATH):
        print("❌ keras_model.h5 NOT FOUND!")
        return
    print("📌 Loading Keras model...")
    model = load_model(MODEL_PATH)
    print("✅ Model Loaded:", model.input_shape)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


def run_clip_inference(clip: np.ndarray):
    """clip shape: (16, 224, 224, 3)"""
    clip = np.expand_dims(clip, axis=0)  # (1,16,224,224,3)
    preds = model.predict(clip)[0]
    non_violence, violence = preds
    label = "Violence" if violence > non_violence else "Non-Violence"
    confidence = float(max(violence, non_violence))
    return {"label": label, "confidence": confidence, "raw": preds.tolist()}


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(None), video_url: str = Form(None)):
    """Predict violence from video upload or URL."""
    if file is None and not video_url:
        return JSONResponse({"error": "Provide a video file or video_url"}, status_code=400)

    temp_path = None

    try:
        # Handle URL download
        if video_url:
            r = requests.get(video_url, stream=True)
            if r.status_code != 200:
                return {"error": "Cannot download video URL"}

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            for chunk in r.iter_content(1024 * 1024):
                temp.write(chunk)
            temp.flush()
            temp_path = temp.name
        else:
            # Upload file
            suffix = os.path.splitext(file.filename)[1]
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp.write(await file.read())
            temp.flush()
            temp_path = temp.name

        # Extract frames
        frames = load_frames_from_video(temp_path)
        if len(frames) < 16:
            return {"error": "Video too short"}

        # Build clips of 16
        clips = build_clips(frames)

        results = [run_clip_inference(c) for c in clips]

        # Final decision: majority vote
        violence_votes = sum(1 for r in results if r["label"] == "Violence")
        final_label = "Violence" if violence_votes > len(results) / 2 else "Non-Violence"

        avg_conf = float(np.mean([r["confidence"] for r in results]))

        return {
            "final_label": final_label,
            "avg_confidence": avg_conf,
            "clips_analyzed": len(results),
            "clip_results": results
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict-webcam")
async def webcam_predict(frames_b64: list = Form(...)):
    """Predict violence from webcam frames (Lovable)."""
    frames = [decode_base64_image(b64) for b64 in frames_b64]
    if len(frames) < 16:
        return {"error": "Need at least 16 frames"}

    clips = build_clips(frames)

    results = [run_clip_inference(c) for c in clips]

    violence_votes = sum(1 for r in results if r["label"] == "Violence")
    final_label = "Violence" if violence_votes > len(results) / 2 else "Non-Violence"
    avg_conf = float(np.mean([r["confidence"] for r in results]))

    return {
        "final_label": final_label,
        "avg_confidence": avg_conf,
        "clips_analyzed": len(results),
        "clip_results": results
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
