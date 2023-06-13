from transformers import pipeline

MODEL_DIR = "model/huggingface_SA_model"


def predict_pipeline(
    text: str, pipeline_name: str = "prediction_pipeline", model_dir: stv = MODEL_DIR
):
    model = pipeline(pipeline_name, model=model_dir)
    prediction = model(text)
    return prediction[0]
