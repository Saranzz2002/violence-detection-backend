import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { AlertCircle, Camera, CameraOff, Shield, AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface WebcamDetectionProps {
  onPrediction?: (result: { isViolence: boolean; confidence: number }) => void;
}

const WebcamDetection = ({ onPrediction }: WebcamDetectionProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [prediction, setPrediction] = useState<{ isViolence: boolean; confidence: number } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const startWebcam = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      setStream(mediaStream);
      setIsActive(true);
      
      // Simulate prediction every 2 seconds
      const interval = setInterval(() => {
        analyzFrame();
      }, 2000);

      return () => clearInterval(interval);
    } catch (error) {
      console.error("Error accessing webcam:", error);
      alert("Unable to access webcam. Please check permissions.");
    }
  };

  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsActive(false);
    setPrediction(null);
  };

  const analyzFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    setIsAnalyzing(true);
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (ctx) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      
      try {
        const imageData = canvas.toDataURL('image/jpeg');
        const PREDICT_URL = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/predict-frame`;
        
        const response = await fetch(PREDICT_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
          },
          body: JSON.stringify({ frame: imageData }),
        });

        const result = await response.json();
        
        setPrediction({
          isViolence: result.isViolence,
          confidence: result.confidence
        });
        setIsAnalyzing(false);
        
        if (onPrediction) {
          onPrediction(result);
        }
      } catch (error) {
        console.error('Error analyzing frame:', error);
        setIsAnalyzing(false);
      }
    }
  };

  return (
    <Card className="p-6 bg-card border-border">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-semibold flex items-center gap-2">
            <Camera className="h-5 w-5 text-primary" />
            Live Webcam Detection
          </h3>
          
          {isActive && (
            <Badge variant={prediction?.isViolence ? "destructive" : "default"} className="animate-pulse">
              {isAnalyzing ? "Analyzing..." : "Monitoring"}
            </Badge>
          )}
        </div>

        <div className="relative bg-secondary rounded-lg overflow-hidden aspect-video flex items-center justify-center">
          {!isActive ? (
            <div className="text-center p-8">
              <Camera className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
              <p className="text-muted-foreground">Click start to begin monitoring</p>
            </div>
          ) : (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover"
              />
              <canvas ref={canvasRef} className="hidden" />
              
              {prediction && (
                <div className={`absolute top-4 right-4 px-4 py-2 rounded-lg backdrop-blur-md ${
                  prediction.isViolence 
                    ? 'bg-destructive/90 border border-destructive' 
                    : 'bg-success/90 border border-success'
                }`}>
                  <div className="flex items-center gap-2">
                    {prediction.isViolence ? (
                      <AlertTriangle className="h-5 w-5 text-white" />
                    ) : (
                      <Shield className="h-5 w-5 text-white" />
                    )}
                    <div className="text-white">
                      <p className="font-bold">
                        {prediction.isViolence ? 'Violence Detected' : 'Safe'}
                      </p>
                      <p className="text-sm opacity-90">
                        {(prediction.confidence * 100).toFixed(1)}% confidence
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex gap-3">
          {!isActive ? (
            <Button onClick={startWebcam} className="flex-1 bg-primary hover:bg-primary/90">
              <Camera className="mr-2 h-4 w-4" />
              Start Monitoring
            </Button>
          ) : (
            <Button onClick={stopWebcam} variant="destructive" className="flex-1">
              <CameraOff className="mr-2 h-4 w-4" />
              Stop Monitoring
            </Button>
          )}
        </div>

        <div className="flex items-start gap-2 p-3 bg-muted/50 rounded-lg">
          <AlertCircle className="h-4 w-4 text-muted-foreground mt-0.5" />
          <p className="text-xs text-muted-foreground">
            Real-time analysis uses 3D CNN with optical flow features. The system monitors continuously and alerts on suspicious activity.
          </p>
        </div>
      </div>
    </Card>
  );
};

export default WebcamDetection;
