# Violence Detection Python Backend

This is the Python/Flask backend for the 3D CNN violence detection model.

## Setup Instructions

### 1. Download Dataset

Download the **RWF-2000** dataset:
```bash
# Visit: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
# Or use direct link from the paper

# Extract to: ./RWF-2000/
# Structure should be:
# RWF-2000/
#   ├── Fight/
#   │   ├── video1.avi
#   │   └── ...
#   └── NonFight/
#       ├── video1.avi
#       └── ...
```

### 2. Install Dependencies

```bash
cd python-backend
pip install -r requirements.txt
```

### 3. Train the Model

```bash
cd model
python train_model.py
```

This will:
- Load the RWF-2000 dataset
- Extract spatial (pixel) and temporal (optical flow) features
- Train the 3D CNN model
- Save the model as `violence_detection_model.h5`

**Note**: Training takes several hours depending on your hardware. The code is set to use 500 samples for demo purposes. Remove the `max_samples` parameter for full training.

### 4. Run the Flask API

```bash
cd api
python app.py
```

The API will run on `http://localhost:5000`

### 5. Deploy to Production

**Option A: Deploy to AWS Lambda**
```bash
# Use Zappa or AWS SAM
pip install zappa
zappa init
zappa deploy production
```

**Option B: Deploy to Google Cloud Run**
```bash
# Create Dockerfile and deploy
gcloud run deploy violence-detection \
  --source . \
  --platform managed \
  --region us-central1
```

**Option C: Deploy to Heroku**
```bash
heroku create violence-detection-api
git push heroku main
```

### 6. Update Frontend

After deploying, copy your API URL and paste it in the Lovable project when prompted for `PYTHON_API_URL`.

## API Endpoints

### POST /api/predict-frame
Analyze a single frame from webcam
```json
{
  "frame": "data:image/jpeg;base64,..."
}
```

### POST /api/analyze-video
Analyze a video from URL
```json
{
  "videoUrl": "F:\Thesis\Project\Realtime-Violence-Detection_Using-DeepLearning-OpenCV-Streamlit\violent_unseen.mp4"
  }
  ```

### GET /health
Health check endpoint

## Model Architecture

- **Dual-stream 3D CNN**
  - Spatial stream: Raw pixel features
  - Temporal stream: Optical flow (Farneback algorithm)
- **Input**: 16 frames at 64x64 resolution
- **Output**: Binary classification (violence/non-violence)

## Dataset Info

**RWF-2000** is recommended for this implementation:
- 2,000 surveillance videos
- 1,000 fight videos
- 1,000 non-fight videos
- Real-world scenarios

Alternative datasets:
- Hockey Fight Dataset (smaller, easier to start)
- UCF-Crime Dataset (larger, more complex)

## Performance Tips

1. **GPU Training**: Use GPU for faster training (CUDA-enabled TensorFlow)
2. **Batch Size**: Adjust based on your GPU memory
3. **Data Augmentation**: Add more training data through augmentation
4. **Fine-tuning**: Experiment with hyperparameters

## Troubleshooting

**Issue**: Model not loading
- Check if `violence_detection_model.h5` exists
- Verify TensorFlow version compatibility

**Issue**: Low accuracy
- Train with full dataset (remove `max_samples` limit)
- Increase epochs (50-100)
- Add data augmentation
- Try different learning rates

**Issue**: Slow inference
- Reduce input resolution
- Use model quantization
- Deploy on GPU instance
