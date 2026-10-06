import os
import cv2
import numpy as np
from model_architecture import create_3d_cnn_model
from feature_extraction import extract_optical_flow, extract_pixel_features

def load_dataset(dataset_path, sequence_length=16, max_samples=None):
    """
    Load RWF-2000 dataset
    Args:
        dataset_path: Path to RWF-2000 dataset (should contain 'Fight' and 'NonFight' folders)
        sequence_length: Number of frames per sequence
        max_samples: Maximum samples to load (None for all)
    Returns:
        X_spatial, X_temporal, y: Training data
    """
    X_spatial = []
    X_temporal = []
    y = []
    
    classes = {'Fight': 1, 'NonFight': 0}
    
    for class_name, label in classes.items():
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist")
            continue
            
        video_files = [f for f in os.listdir(class_path) if f.endswith('.avi')]
        
        if max_samples:
            video_files = video_files[:max_samples]
        
        print(f"Loading {len(video_files)} videos from {class_name}...")
        
        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            
            # Read video
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while len(frames) < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if len(frames) < sequence_length:
                continue
            
            # Extract features
            spatial = extract_pixel_features(frames[:sequence_length])
            temporal = extract_optical_flow(frames[:sequence_length])
            
            X_spatial.append(spatial)
            X_temporal.append(temporal)
            y.append(label)
    
    return np.array(X_spatial), np.array(X_temporal), np.array(y)

def train(dataset_path, epochs=50, batch_size=8):
    """
    Train the 3D CNN model
    """
    print("Loading dataset...")
    X_spatial, X_temporal, y = load_dataset(
        dataset_path, 
        sequence_length=16,
        max_samples=500  # Limit for demo - remove for full training
    )
    
    print(f"Dataset loaded: {len(y)} samples")
    print(f"Spatial shape: {X_spatial.shape}")
    print(f"Temporal shape: {X_temporal.shape}")
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_spatial_train = X_spatial[train_idx]
    X_temporal_train = X_temporal[train_idx]
    y_train = y[train_idx]
    
    X_spatial_val = X_spatial[val_idx]
    X_temporal_val = X_temporal[val_idx]
    y_val = y[val_idx]
    
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    
    # Create model
    print("Creating model...")
    model = create_3d_cnn_model(
        input_shape=X_spatial_train.shape[1:],
        flow_shape=X_temporal_train.shape[1:]
    )
    
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        [X_spatial_train, X_temporal_train],
        y_train,
        validation_data=([X_spatial_val, X_temporal_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Save model
    model.save('violence_detection_model.h5')
    print("Model saved as 'violence_detection_model.h5'")
    
    return model, history

if __name__ == "__main__":
    # Download RWF-2000 dataset first from:
    # https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
    
    DATASET_PATH = "./RWF-2000"  # Update this path
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please download RWF-2000 dataset first")
    else:
        model, history = train(DATASET_PATH, epochs=50, batch_size=8)
