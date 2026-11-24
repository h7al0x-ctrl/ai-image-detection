"""
The ONE metric implementation used for all three models.

Positive class = AI-generated (1). Predictions use threshold 0.5 on the
probability of class 1. ROC-AUC uses the continuous probabilities.
Undefined values are returned as NaN and explained in the `notes` field:
  * precision undefined  -> the model predicted no AI images (tp + fp = 0)
  * recall undefined     -> the subset has no AI images
  * ROC-AUC undefined    -> the subset contains only one class
"""
import math

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

import config

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"]


def _safe_div(a, b):
    return a / b if b else float("nan")


def compute_metrics(y_true, y_prob, threshold=config.THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("labels must be 0 (real) or 1 (AI-generated)")
    if not np.all(np.isfinite(y_prob)) or y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob must be probabilities in [0, 1]")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    notes = []

    precision = _safe_div(tp, tp + fp)
    if math.isnan(precision):
        notes.append("precision undefined: no AI predictions")
    recall = _safe_div(tp, tp + fn)
    if math.isnan(recall):
        notes.append("recall undefined: no AI images")
    specificity = _safe_div(tn, tn + fp)
    if math.isnan(specificity):
        notes.append("specificity undefined: no real images")
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    balanced = (recall + specificity) / 2 if not (math.isnan(recall) or math.isnan(specificity)) else float("nan")
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = float("nan")
        notes.append("ROC-AUC undefined: only one class present")

    return {
        "n": int(len(y_true)), "n_real": int((y_true == 0).sum()), "n_ai": int((y_true == 1).sum()),
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else float("nan"),
        "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc,
        "balanced_accuracy": balanced, "specificity": specificity,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold, "notes": "; ".join(notes),
    }


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == 1 and m["roc_auc"] == 1 and m["f1"] == 1
    m = compute_metrics(y, [0.9, 0.8, 0.2, 0.1])
    assert m["accuracy"] == 0 and m["roc_auc"] == 0
    m = compute_metrics(y, [0.1, 0.1, 0.1, 0.1])
    assert math.isnan(m["precision"]) and m["recall"] == 0 and m["balanced_accuracy"] == 0.5
    m = compute_metrics([1, 1], [0.7, 0.4])
    assert math.isnan(m["roc_auc"]) and m["recall"] == 0.5
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score
    rng = np.random.default_rng(42)
    yt = rng.integers(0, 2, 500)
    yp = rng.random(500)
    m = compute_metrics(yt, yp)
    assert abs(m["f1"] - f1_score(yt, yp >= 0.5)) < 1e-12
    assert abs(m["precision"] - precision_score(yt, yp >= 0.5)) < 1e-12
    assert abs(m["balanced_accuracy"] - balanced_accuracy_score(yt, yp >= 0.5)) < 1e-12
    print("metrics self-test passed")
