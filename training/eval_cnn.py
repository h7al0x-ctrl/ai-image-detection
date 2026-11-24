"""
CNN evaluation for every entry of config.PREDICTION_PLAN
(Experiments 1-3, robustness conditions, 32x32). Saved models are only LOADED:
no retraining, no fine-tuning, fixed threshold 0.5.

Output: results/predictions/cnn_predictions.csv

Run:  python -m training.eval_cnn
"""
import numpy as np
import torch

import config
from evaluation.predictions import build_prediction_frame, save_predictions
from models.cnn_model import load_cnn
from preprocessing.image_cache import make_loader
from training.train_cnn import checkpoint_path
from utils import ensure_dirs, get_device, log_run, set_seed


@torch.no_grad()
def predict_probs(model, loader, device):
    probs = []
    for x, _ in loader:
        probs.append(model.predict_proba(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(probs)


def main():
    ensure_dirs()
    set_seed()
    device = get_device()
    frames, models = [], {}
    for entry in config.PREDICTION_PLAN:
        tag = entry["model_tag"]
        if tag not in models:
            path = checkpoint_path(tag)
            if not path.exists():
                print(f"WARNING: {path.name} missing - skipping its experiments")
                models[tag] = None
                continue
            models[tag], ckpt = load_cnn(path, device)
            print(f"loaded {path.name} (best epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f})")
        if models[tag] is None:
            continue
        loader, manifest = make_loader(entry["dataset"], entry["split"], entry["condition"], train=False)
        probs = predict_probs(models[tag], loader, device)
        frames.append(build_prediction_frame("cnn", entry, manifest, probs))
        acc = (frames[-1]["pred"] == frames[-1]["label"]).mean()
        print(f"  {entry['experiment']:6s} {entry['condition']:10s} accuracy={acc:.4f}")
    if frames:
        df = save_predictions("cnn", frames)
        log_run("student5_cnn", "evaluate", n_rows=len(df))


if __name__ == "__main__":
    main()
