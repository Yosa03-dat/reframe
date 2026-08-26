from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os

from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.inference import InferencePipeline

# Global reference for the pipeline
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    
    classifier_path = os.path.join("results", "classifier", "best_model")
    generator_path = os.path.join("results", "generator", "best_model")
    
    if not os.path.exists(classifier_path) or not os.path.exists(generator_path):
        raise RuntimeError(f"Trained models not found at {classifier_path} or {generator_path}. Please train the models first.")
        
    print("Loading models asynchronously during startup...")
    pipeline = InferencePipeline(classifier_path, generator_path)
    
    yield
    
    # Cleanup on shutdown if needed
    pipeline = None

app = FastAPI(
    title="Reframe API",
    description="API for detecting toxic speech and generating constructive interventions.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        # The inference pipeline is synchronous but heavily GPU optimized.
        # For a truly async IO bound app, this could be run in a threadpool, 
        # but for PyTorch on GPU it's fine for our usecase.
        result = pipeline.analyze(request.text)
        return AnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
