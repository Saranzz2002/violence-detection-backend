import cv2
import numpy as np

def extract_optical_flow(frames):
    """
    Extract optical flow features using Farneback algorithm
    Args:
        frames: List of video frames
    Returns:
        optical_flow: Array of optical flow features
    """
    flow_features = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Convert to magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_features.append(np.stack([magnitude, angle], axis=-1))
        
        prev_gray = gray
    
    return np.array(flow_features)

def extract_pixel_features(frames, target_size=(64, 64)):
    """
    Extract raw pixel features from frames
    Args:
        frames: List of video frames
        target_size: Resize dimensions
    Returns:
        pixel_features: Array of normalized pixel features
    """
    pixel_features = []
    
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, target_size)
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        pixel_features.append(normalized)
    
    return np.array(pixel_features)

def preprocess_frame(frame_base64, target_size=(64, 64)):
    """
    Preprocess a single frame from base64 for prediction
    """
    import base64
    from io import BytesIO
    from PIL import Image
    
    # Decode base64
    image_data = base64.b64decode(frame_base64.split(',')[1])
    image = Image.open(BytesIO(image_data))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Resize and normalize
    resized = cv2.resize(frame, target_size)
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized
