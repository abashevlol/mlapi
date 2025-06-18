from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

# Load the ML model once at startup
try:
    model = joblib.load("./iris.mdl")
except Exception as e:
    model = None
    print(f"Model loading failed: {e}")

# Pydantic models
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class NameRequest(BaseModel):
    name: str

# Single FastAPI app instance
app = FastAPI(
    title="Iris & Hello API",
    description="Predict the species of an iris flower or get a personalized greeting.",
    version="2.0.0"
)

@app.post("/predict")
async def predict(
    features: IrisFeatures,
    x_api_token: str = Header(..., alias="X-API-Token")
):
    """
    Predict the Iris species from sepal and petal measurements.
    Requires a valid X-API-Token header.
    """
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_df = pd.DataFrame([features.dict()])
        prediction = model.predict(input_df)
        return {"predicted_species": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/hello", summary="Greet the user", response_description="A greeting message")
def hello(data: NameRequest):
    """
    Returns a personalized greeting message.
    """
    return {"message": f"Hello {data.name}"}
