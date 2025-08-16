from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load model and target names
model = joblib.load("model.pkl")
target_names = joblib.load("target_names.pkl")

app = FastAPI(title="Iris Classifier API", description="Predict Iris species")

# Enable CORS just in case
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output schema
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

# Serve HTML front-end
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_file = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(html_file, "r") as f:
        return f.read()

# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

# Predict endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array([[input_data.sepal_length,
                              input_data.sepal_width,
                              input_data.petal_length,
                              input_data.petal_width]])
        proba = model.predict_proba(features)[0]
        pred_index = np.argmax(proba)
        return PredictionOutput(prediction=target_names[pred_index],
                                confidence=float(proba[pred_index]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Model info
@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": target_names.tolist()
    }