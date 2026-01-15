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

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    target_frames = 128
    if mel_db.shape[1] < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :target_frames]

    x = mel_db.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)

    x = np.expand_dims(x, axis=0)   # batch
    x = np.expand_dims(x, axis=-1)  # channel
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
