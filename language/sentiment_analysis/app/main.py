from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_pipeline, load_model

app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    Sentiment: str
    Score: float | None = None


MODEL_DIR = "/app/app/model/huggingface_SA_model_0.0.1"
load_model(model_dir=MODEL_DIR)


@app.get("/")
def home():
    return {"Health": "OK"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: TextIn):
    prediction = predict_pipeline(payload.text)
    return {"Sentiment": prediction["label"]}


@app.post("/predict_proba", response_model=PredictionOutput)
def predict_proba(payload: TextIn):
    prediction = predict_pipeline(payload.text)
    return {"Sentiment": prediction["label"], "Score": prediction["score"]}
