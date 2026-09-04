"""Deterministic inference with the FROZEN EfficientNetB0 (SIH26038 Phase 5).

Same preprocessing as training/inference baseline: PIL RGB resize to
image_size + efficientnet.preprocess_input, batched model.predict.
Weights are loaded read-only; nothing is re-saved or modified.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

from src.evaluation.types import PredictionRecord

_model_cache = {}


def get_model(model_path):
    key = str(model_path)
    if key not in _model_cache:
        _model_cache[key] = load_model(key)
    return _model_cache[key]


def preprocess_image(path, image_size=224):
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return preprocess_input(np.array(img).astype("float32"))


def run_inference(df, model_path, image_size=224, batch_size=32, split="test"):
    """df rows: id_code, diagnosis, image_path. Returns [PredictionRecord]."""
    model = get_model(model_path)
    records = []
    paths = list(df["image_path"])
    for i in range(0, len(paths), batch_size):
        chunk = df.iloc[i:i + batch_size]
        batch = np.stack([preprocess_image(p, image_size) for p in chunk["image_path"]])
        probs = model.predict(batch, verbose=0)
        for (_, row), pr in zip(chunk.iterrows(), probs):
            pr = np.asarray(pr, dtype=float)
            records.append(PredictionRecord(
                image_id=row["id_code"], ground_truth=int(row["diagnosis"]),
                predicted_grade=int(np.argmax(pr)), probabilities=list(pr),
                raw_confidence=float(np.max(pr)), split=split))
    return records
