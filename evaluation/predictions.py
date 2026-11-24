"""Shared prediction-file format (one CSV per model in results/predictions/)."""
import numpy as np
import pandas as pd

import config

PRED_COLUMNS = ["model", "experiment", "condition", "model_tag", "sample_id", "image_path",
                "dataset", "split", "source_group", "generator", "label", "prob", "pred"]


def build_prediction_frame(model_name, entry, manifest_df, probs, sample_ids=None):
    """Combine manifest rows and predicted probabilities for one plan entry."""
    probs = np.asarray(probs, dtype=float)
    if len(probs) != len(manifest_df):
        raise ValueError("number of probabilities does not match manifest rows")
    if sample_ids is not None and not np.array_equal(np.asarray(sample_ids, dtype=str), manifest_df["sample_id"].to_numpy(dtype=str)):
        raise ValueError("sample order mismatch between features/predictions and manifest")
    df = manifest_df[["sample_id", "image_path", "dataset", "split", "source_group", "generator", "label"]].copy()
    df.insert(0, "model", model_name)
    df.insert(1, "experiment", entry["experiment"])
    df.insert(2, "condition", entry["condition"])
    df.insert(3, "model_tag", entry["model_tag"])
    df["prob"] = probs
    df["pred"] = (probs >= config.THRESHOLD).astype(int)
    return df[PRED_COLUMNS]


def prediction_path(model_name):
    return config.PRED_DIR / f"{model_name}_predictions.csv"


def save_predictions(model_name, frames):
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(prediction_path(model_name), index=False)
    print(f"saved {len(df)} predictions -> {prediction_path(model_name)}")
    return df


def load_all_predictions():
    frames = [pd.read_csv(prediction_path(m)) for m in config.MODELS if prediction_path(m).exists()]
    if not frames:
        raise FileNotFoundError("no prediction files found in results/predictions/")
    return pd.concat(frames, ignore_index=True)
