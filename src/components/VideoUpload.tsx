import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Link2, Upload, AlertTriangle, Shield, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const VideoUpload = () => {
  const [videoUrl, setVideoUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{ isViolence: boolean; confidence: number; timestamp?: string } | null>(null);

  const analyzeVideo = async () => {
    if (!videoUrl.trim()) {
      alert("Please enter a video URL");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const ANALYZE_URL = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/analyze-video`;
      
      const response = await fetch(ANALYZE_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({ videoUrl }),
      });

      const data = await response.json();
      
      const analysisResult = {
        isViolence: data.isViolence,
        confidence: data.confidence,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setResult(analysisResult);
      setIsAnalyzing(false);
    } catch (error) {
      console.error('Error analyzing video:', error);
      alert('Failed to analyze video. Please check if the Python backend is running.');
      setIsAnalyzing(false);
    }
  };

  return (
    <Card className="p-6 bg-card border-border">
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Link2 className="h-5 w-5 text-primary" />
          <h3 className="text-xl font-semibold">Video Link Analysis</h3>
        </div>

        <div className="space-y-3">
          <div className="flex gap-2">
            <Input
              type="url"
              placeholder="Enter video URL (YouTube, Vimeo, direct link...)"
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              className="flex-1 bg-secondary border-border"
              disabled={isAnalyzing}
            />
          </div>

          <Button 
            onClick={analyzeVideo} 
            disabled={isAnalyzing || !videoUrl.trim()}
            className="w-full bg-primary hover:bg-primary/90"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing Video...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Analyze Video
              </>
            )}
          </Button>
        </div>

        {isAnalyzing && (
          <div className="p-4 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <div className="space-y-1">
                <p className="text-sm font-medium">Processing video...</p>
                <p className="text-xs text-muted-foreground">
                  Extracting frames, computing optical flow, and analyzing with 3D CNN
                </p>
              </div>
            </div>
            
            <div className="mt-3 space-y-1.5">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Frame extraction</span>
                <span>100%</span>
              </div>
              <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full animate-pulse w-full" />
              </div>
            </div>
          </div>
        )}

        {result && !isAnalyzing && (
          <div className={`p-6 rounded-lg border-2 ${
            result.isViolence 
              ? 'bg-destructive/10 border-destructive' 
              : 'bg-success/10 border-success'
          }`}>
            <div className="flex items-start gap-4">
              <div className={`p-3 rounded-full ${
                result.isViolence ? 'bg-destructive' : 'bg-success'
              }`}>
                {result.isViolence ? (
                  <AlertTriangle className="h-6 w-6 text-white" />
                ) : (
                  <Shield className="h-6 w-6 text-white" />
                )}
              </div>
              
              <div className="flex-1">
                <h4 className={`text-lg font-bold mb-1 ${
                  result.isViolence ? 'text-destructive' : 'text-success'
                }`}>
                  {result.isViolence ? 'Violence Detected' : 'No Violence Detected'}
                </h4>
                
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">Confidence:</span>
                    <Badge variant={result.isViolence ? "destructive" : "default"}>
                      {(result.confidence * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  
                  <p className="text-xs text-muted-foreground">
                    Analyzed at {result.timestamp}
                  </p>
                  
                  {result.isViolence && (
                    <p className="text-sm mt-3 p-3 bg-background/50 rounded border border-border">
                      ⚠️ Violent activity detected in video. This content may require review by authorities.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="p-3 bg-muted/50 rounded-lg space-y-2">
          <p className="text-xs font-medium text-foreground">Supported formats:</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary" className="text-xs">YouTube</Badge>
            <Badge variant="secondary" className="text-xs">Vimeo</Badge>
            <Badge variant="secondary" className="text-xs">MP4</Badge>
            <Badge variant="secondary" className="text-xs">AVI</Badge>
            <Badge variant="secondary" className="text-xs">MOV</Badge>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default VideoUpload;
