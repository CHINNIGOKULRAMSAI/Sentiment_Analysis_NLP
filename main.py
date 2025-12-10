import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts whether a review is POSITIVE or NEGATIVE",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=FileResponse)
def serve_home():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))


class TextRequest(BaseModel):
    text: str


predict_pipeline = PredictPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_sentiment(request: TextRequest):
    text = (request.text or "").strip()

    if not text:
        return JSONResponse(
            status_code=400,
            content={
                "label": "Unknown",
                "confidence": None,
                "error": "Empty text",
            },
        )

    preds = predict_pipeline.Predict(text)
    pred = int(preds[0])

    confidence = None
    if hasattr(predict_pipeline.model, "predict_proba"):
        cleaned = predict_pipeline.dt._clean_text(text)
        vec = predict_pipeline.preprocessor.transform([cleaned])
        proba = predict_pipeline.model.predict_proba(vec)[0]
        confidence = float(max(proba))

    label = "😡 NEGATIVE" if pred == 0 else "😄 POSITIVE"

    return {
        "label": label,
        "prediction": pred,
        "confidence": confidence,
        "text": text,
    }
