from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.feature_extraction import preprocess_frame, extract_optical_flow, extract_pixel_features
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load model at startup
MODEL_PATH = 'violence_detection_model.h5'
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Using mock predictions. Train the model first!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using mock predictions")

@app.route('/api/predict-frame', methods=['POST'])
def predict_frame():
    """
    Predict violence in a single webcam frame
    """
    try:
        data = request.json
        frame_base64 = data.get('frame')
        
        if not frame_base64:
            return jsonify({'error': 'No frame provided'}), 400
        
        # For real prediction, you'd need multiple frames for temporal analysis
        # This is a simplified version for single frame
        
        if model is None:
            # Mock prediction
            confidence = np.random.random() * 0.4 + 0.6
            is_violence = np.random.random() > 0.7
            
            return jsonify({
                'isViolence': bool(is_violence),
                'confidence': float(confidence),
                'note': 'Using mock prediction - train model for real predictions'
            })
        
        # Preprocess frame
        frame = preprocess_frame(frame_base64)
        
        # Note: Real implementation needs sequence of frames
        # This is a simplified demo
        frames = np.expand_dims(frame, axis=0)
        frames = np.repeat(frames, 16, axis=0)  # Duplicate to create sequence
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        
        # Create dummy temporal features
        temporal = np.random.random((1, 16, 64, 64, 2)).astype(np.float32)
        
        # Predict
        prediction = model.predict([frames, temporal], verbose=0)
        confidence = float(prediction[0][0])
        is_violence = confidence > 0.5
        
        return jsonify({
            'isViolence': bool(is_violence),
            'confidence': float(confidence)
        })
        
    except Exception as e:
        print(f"Error in predict-frame: {e}")
        return jsonify({
            'error': str(e),
            'isViolence': False,
            'confidence': 0.0
        }), 500

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """
    Analyze violence in a video URL
    """
    try:
        data = request.json
        video_url = data.get('videoUrl')
        
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400
        
        # Download and process video
        import requests
        from io import BytesIO
        
        # For demo: Mock prediction
        if model is None:
            confidence = np.random.random() * 0.3 + 0.7
            is_violence = np.random.random() > 0.6
            
            return jsonify({
                'isViolence': bool(is_violence),
                'confidence': float(confidence),
                'framesAnalyzed': 150,
                'note': 'Using mock prediction - train model for real predictions'
            })
        
        # Real implementation would download and process video
        # For now, return mock data
        
        return jsonify({
            'isViolence': False,
            'confidence': 0.3,
            'framesAnalyzed': 0,
            'error': 'Video analysis not fully implemented yet'
        })
        
    except Exception as e:
        print(f"Error in analyze-video: {e}")
        return jsonify({
            'error': str(e),
            'isViolence': False,
            'confidence': 0.0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
