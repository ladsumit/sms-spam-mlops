# SMS Spam Detection with MLflow and FastAPI

## Overview
This project demonstrates a simple MLOps workflow using MLflow for experiment tracking and model management, combined with FastAPI for model serving. It trains a logistic regression classifier to detect SMS spam messages.

## Dataset
Place your dataset at:
data/raw/sms_spam.csv

It should contain two columns:
- text — message content
- target — labels ("spam" or "ham")

Example:
target,text
spam,WIN a FREE prize!!! Reply WIN
ham,Hey are we still on for lunch?

## Training the Model
Run the MLflow project:
mlflow run . --experiment-name "SMS_Spam_Quickstart"

This trains the model, evaluates performance, logs metrics, and saves the model artifact to mlruns/.

## Viewing Results
Launch the MLflow UI to inspect runs:
mlflow ui

Visit http://127.0.0.1:5000 to view experiments, metrics, and artifacts.

## Serving the Model with FastAPI
Activate your serving environment and install dependencies:
python -m venv .serve-venv
source .serve-venv/bin/activate
pip install -r model_requirements.txt fastapi uvicorn

Export your trained model’s URI (replace with your run ID):
export MODEL_URI="runs:/<your_run_id>/model"

Start the FastAPI server:
python -m uvicorn app_sms:app --host 0.0.0.0 --port 8000 --reload

## Testing the API
Health check:
curl -s http://127.0.0.1:8000/healthz

Prediction:
curl -s -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"messages":["WIN a FREE prize!!! Reply WIN","hey are we still on for lunch?"]}'

Expected response:
{"predictions":[1,0],"labels":["spam","ham"]}

## Notes
- Always use python -m uvicorn from the .serve-venv to ensure correct dependencies.
- The MLflow project logs dependencies automatically during training.
- You can generate an environment-specific model_requirements.txt via:
  python - <<'PY'
  import os, mlflow, pathlib
  req = mlflow.pyfunc.get_model_dependencies(os.environ["MODEL_URI"])
  pathlib.Path("model_requirements.txt").write_text(open(req).read())
  PY
