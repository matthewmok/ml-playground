from transformers import pipeline
import os

model = None


def predict_pipeline(text: str):
    prediction = model(text)
    return prediction[0]


def load_model(model_dir: str):
    global model
    print(f"Loading model from {model_dir}")
    model = pipeline("sentiment-analysis", model=model_dir)
