"""
SVM on ResNet50 features.

Pipeline = StandardScaler (fitted on TRAIN only, inside the pipeline) + SVC.
Hyper-parameters (config.SVM_PARAM_GRID) are chosen on the validation split.
The final model uses probability=True (Platt scaling, fitted on train data by
internal cross-validation) so that it outputs probabilities for threshold 0.5 and ROC-AUC.

Outputs: checkpoints/svm/svm_{cifake,genimage,res32}.joblib,
         results/tables/svm_<tag>_tuning.csv, results/predictions/svm_predictions.csv

Run:  python -m models.svm_model                 (all model tags)
      python -m models.svm_model --tags cifake   (one tag)
"""
import argparse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from models.feature_model_common import run
from utils import ensure_dirs, set_seed


def make_svm(params, final=False):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**params, probability=final, random_state=config.SEED, cache_size=config.SVM_CACHE_MB)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="*", choices=list(config.MODEL_TAGS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    set_seed()
    run("svm", make_svm, config.SVM_PARAM_GRID, args.tags, args.force)


if __name__ == "__main__":
    main()
