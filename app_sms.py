# sms_spam/app_sms.py
import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

MODEL_URI = os.environ.get("MODEL_URI")
if not MODEL_URI:
    raise RuntimeError("Set MODEL_URI to an MLflow model like runs:/<run_id>/model")

app = FastAPI(title="SMS Spam Classifier")

# Load at startup (kept simple; you can move to startup event if you prefer)
model = mlflow.pyfunc.load_model(MODEL_URI)
LABELS = {0: "ham", 1: "spam"}

class PredictTextRequest(BaseModel):
    messages: List[str] = Field(..., description="List of SMS or short texts.")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post("/predict")
def predict(req: PredictTextRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    preds = model.predict(pd.Series(req.messages)).tolist()
    labels = [LABELS.get(int(p), str(p)) for p in preds]
    return {"predictions": preds, "labels": labels}

@app.post("/predict_proba")
def predict_proba(req: PredictTextRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    # scikit pipelines expose predict_proba
    proba = model.predict_proba(pd.Series(req.messages)).tolist()
    # return spam probability as convenience
    spam_prob = [float(p[1]) for p in proba]
    return {"probabilities": proba, "spam_probability": spam_prob}
