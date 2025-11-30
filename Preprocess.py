import cv2
import numpy as np
from typing import List
from PIL import Image
import base64
import io

TARGET_H = 224
TARGET_W = 224
CLIP_LEN = 16


def load_frames_from_video(path: str, max_frames: int = 500) -> List[np.ndarray]:
    """Read frames from a video file (BGR)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode base64 (data:image/...;base64,XXX) to BGR frame."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1]  # RGB → BGR


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to model input shape."""
    resized = cv2.resize(frame, (TARGET_W, TARGET_H))
    resized = resized.astype("float32") / 255.0
    return resized


def build_clips(frames: List[np.ndarray], clip_len: int = CLIP_LEN):
    """Convert list of frames into list of clips of size 16."""
    clips = []
    total = len(frames)
    for i in range(0, total - clip_len + 1, clip_len):
        chunk = frames[i: i + clip_len]
        chunk = np.array([preprocess_frame(f) for f in chunk])
        clips.append(chunk)
    return clips
