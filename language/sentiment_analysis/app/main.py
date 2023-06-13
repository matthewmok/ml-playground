from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_pipeline

app = FastAPI()

class PredictionOutPut(BaseModel):
    Sentiment: str

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post("/predict", response_model=PredictionOutput):
def predict(payload: TextIn):
    prediction = predict_pipeline(payload.text)
    return {"Sentiment": prediction['label']}

@app.post("/predict_proba", response_model=PredictionOutput):
def predict_proba(payload: TextIn):
    prediction = predict_pipeline(payload.text)
    return {"Sentiment": prediction['label'], "Score": prediction['score']}

