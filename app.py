from fastapi import FastAPI
from pydantic import BaseModel
import requests, tempfile, os

import numpy as np
import librosa
import tensorflow as tf

app = FastAPI()

MODEL_PATH = "cough_sound_classification.hdf5"
model = tf.keras.models.load_model(MODEL_PATH)

class AnalyzeRequest(BaseModel):
    audio_url: str

def preprocess_audio(path: str):
    y, sr = librosa.load(path, sr=22050)

    # Compute MFCCs (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Aggregate over time to get fixed-length vector
    mfcc_mean = np.mean(mfcc, axis=1)  # shape (40,)

    x = mfcc_mean.astype(np.float32)

    # Optional normalization
    x = (x - x.mean()) / (x.std() + 1e-6)

    x = np.expand_dims(x, axis=0)  # shape (1, 40)
    return x


@app.get("/")
def root():
    return {"status": "ok", "message": "Cough analyzer API is running"}

@app.post("/analyze-cough")
def analyze(req: AnalyzeRequest):
    r = requests.get(req.audio_url, timeout=30)
    r.raise_for_status()

    # For now we assume the file bytes represent a WAV-like audio.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(r.content)
        tmp_path = f.name

    try:
        x = preprocess_audio(tmp_path)
        pred = model.predict(x)[0]

        # Binary or 2-class softmax
        if pred.shape[0] == 1:
            score = float(pred[0])
            label = "wet" if score >= 0.5 else "dry"
            confidence = score if label == "wet" else 1 - score
        else:
            idx = int(np.argmax(pred))
            confidence = float(pred[idx])
            label = ["dry", "wet"][idx]  # swap if results look inverted

        return {"label": label, "confidence": round(confidence, 4)}

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
