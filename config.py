"""
Central configuration for the AI-Generated Image Detection project.

EVERY shared setting lives here. Do not change a value without team agreement,
and record the change in README.md ("Decision log").

Paths can be overridden with environment variables (used by the Colab notebook):
    AIID_CIFAKE_ROOT, AIID_GENIMAGE_ROOT, AIID_CACHE_DIR, AIID_FEATURE_DIR
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root unless overridden)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(var_name, default):
    value = os.environ.get(var_name)
    return Path(value).expanduser() if value else Path(default)


DATA_DIR = PROJECT_ROOT / "data"
CIFAKE_ROOT = _path_from_env("AIID_CIFAKE_ROOT", DATA_DIR / "CIFAKE")
GENIMAGE_ROOT = _path_from_env("AIID_GENIMAGE_ROOT", DATA_DIR / "GenImage")
MANIFEST_DIR = DATA_DIR / "manifests"
# Preprocessed 224x224 uint8 image arrays (.npy). Large -> keep on local disk.
CACHE_DIR = _path_from_env("AIID_CACHE_DIR", DATA_DIR / "cache")
FEATURE_DIR = _path_from_env("AIID_FEATURE_DIR", PROJECT_ROOT / "features" / "saved")

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
LOG_DIR = PROJECT_ROOT / "logs"
RUN_LOG = LOG_DIR / "run_log.csv"

OUTPUT_DIRS = [
    MANIFEST_DIR, CACHE_DIR, FEATURE_DIR,
    CHECKPOINT_DIR / "svm", CHECKPOINT_DIR / "rf", CHECKPOINT_DIR / "cnn",
    PRED_DIR, TABLE_DIR, FIGURE_DIR, ROBUSTNESS_DIR, LOG_DIR,
]

# ---------------------------------------------------------------------------
# Agreed global settings
# ---------------------------------------------------------------------------
SEED = 42
LABEL_REAL = 0
LABEL_AI = 1
CLASS_NAMES = {LABEL_REAL: "Real", LABEL_AI: "AI-generated"}
POSITIVE_CLASS = LABEL_AI
THRESHOLD = 0.5

IMAGE_SIZE = 224
COLOR_MODE = "RGB"
RESIZE_INTERPOLATION = "bicubic"
RESIZE_POLICY = "shorter side -> 224 (bicubic), then center crop 224x224"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

BATCH_SIZE = 32
NUM_WORKERS = 2  # Colab has 2 CPU cores

BACKBONE = "resnet50"
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
FEATURE_DIM = 2048

# ---------------------------------------------------------------------------
# Datasets and sample counts (decided by the team: 20,000 images per dataset)
# ---------------------------------------------------------------------------
CIFAKE = "CIFAKE"
GENIMAGE = "GenImage"
DATASETS = [CIFAKE, GENIMAGE]
SPLITS = ["train", "val", "test"]

# CIFAKE: images PER CLASS. Keys are the OFFICIAL CIFAKE split folders,
# values are the project splits taken from them (train/val never touch official test).
CIFAKE_PER_CLASS = {
    "train": {"train": 7000, "val": 1000},
    "test": {"test": 2000},
}
CIFAKE_JPEG_QUALITY = None  # CIFAKE is used as provided

# GenImage generators
GENIMAGE_TRAIN_GENERATORS = ["ADM", "GLIDE", "SD15", "VQDM", "Wukong"]
GENIMAGE_TEST_GENERATORS = ["BigGAN", "Midjourney"]  # held-out: TEST ONLY
GENIMAGE_ALL_GENERATORS = GENIMAGE_TRAIN_GENERATORS + GENIMAGE_TEST_GENERATORS

# Images PER CLASS per generator (2,500 images per generator in total)
GENIMAGE_PER_CLASS_TRAIN_GEN = {"train": 1000, "val": 250}
GENIMAGE_PER_CLASS_TEST_GEN = {"test": 1250}

# Every selected GenImage image (real AND AI) is re-encoded as JPEG Q95 BEFORE
# resizing, so that the file format (JPEG real vs PNG fake) cannot be a shortcut.
GENIMAGE_JPEG_QUALITY = 95

# Keywords used to recognise generator folders (case-insensitive substring match).
# Covers the official names, e.g. "imagenet_ai_0419_sdv4", "imagenet_midjourney".
GENIMAGE_FOLDER_KEYWORDS = {
    "ADM": ["adm"],
    "GLIDE": ["glide"],
    "SD14": ["sdv4", "sd14", "sd_1_4", "sd1.4", "sdv1.4", "sdv14", "stable_diffusion_v_1_4"],
    "SD15": ["sdv5", "sd15", "sd_1_5", "sd1.5", "sdv1.5", "sdv15", "stable_diffusion_v_1_5"],
    "VQDM": ["vqdm"],
    "Wukong": ["wukong"],
    "BigGAN": ["biggan"],
    "Midjourney": ["midjourney"],
}
GENIMAGE_AI_DIR_NAMES = ("ai", "fake")
GENIMAGE_REAL_DIR_NAMES = ("nature", "real")

# Perceptual-hash near-duplicate threshold (Hamming distance on 64-bit pHash).
PHASH_MAX_DISTANCE = 2


def expected_counts(dataset):
    """Return {split: {label: count}} expected for a dataset."""
    out = {s: {LABEL_REAL: 0, LABEL_AI: 0} for s in SPLITS}
    if dataset == CIFAKE:
        for _, project_splits in CIFAKE_PER_CLASS.items():
            for split, n in project_splits.items():
                out[split][LABEL_REAL] += n
                out[split][LABEL_AI] += n
    else:
        for gen in GENIMAGE_ALL_GENERATORS:
            per = GENIMAGE_PER_CLASS_TEST_GEN if gen in GENIMAGE_TEST_GENERATORS else GENIMAGE_PER_CLASS_TRAIN_GEN
            for split, n in per.items():
                out[split][LABEL_REAL] += n
                out[split][LABEL_AI] += n
    return out


# ---------------------------------------------------------------------------
# Image conditions (robustness transformations + the 32x32 condition).
# All are applied to the cached 224x224 RGB image, BEFORE ImageNet normalisation.
# ---------------------------------------------------------------------------
CONDITIONS = {
    "clean": {"type": "none", "label": "Clean baseline", "params": "none"},
    "jpeg90": {"type": "jpeg", "quality": 90, "label": "JPEG Q90", "params": "quality=90"},
    "jpeg70": {"type": "jpeg", "quality": 70, "label": "JPEG Q70", "params": "quality=70"},
    "jpeg50": {"type": "jpeg", "quality": 50, "label": "JPEG Q50", "params": "quality=50"},
    "resize112": {"type": "resize", "size": 112, "label": "Resize 112", "params": "224->112->224 bicubic"},
    "resize56": {"type": "resize", "size": 56, "label": "Resize 56", "params": "224->56->224 bicubic"},
    "noise001": {"type": "noise", "std": 0.01, "label": "Noise 0.01", "params": "sigma=0.01 on [0,1], clipped"},
    "noise003": {"type": "noise", "std": 0.03, "label": "Noise 0.03", "params": "sigma=0.03 on [0,1], clipped"},
    "noise005": {"type": "noise", "std": 0.05, "label": "Noise 0.05", "params": "sigma=0.05 on [0,1], clipped"},
    # Additional resolution experiment (train AND test under this condition)
    "res32": {"type": "resize", "size": 32, "label": "32x32 resolution", "params": "224->32->224 bicubic"},
}
ROBUSTNESS_CONDITIONS = ["clean", "jpeg90", "jpeg70", "jpeg50", "resize112", "resize56",
                         "noise001", "noise003", "noise005"]

# ---------------------------------------------------------------------------
# Models, training sets and experiments
# ---------------------------------------------------------------------------
MODELS = ["svm", "rf", "cnn"]
MODEL_DISPLAY = {"svm": "SVM", "rf": "Random Forest", "cnn": "CNN (ResNet50)"}

# A "model tag" = which data a model is trained/validated on.
MODEL_TAGS = {
    "cifake": {"dataset": CIFAKE, "condition": "clean"},
    "genimage": {"dataset": GENIMAGE, "condition": "clean"},
    "res32": {"dataset": GENIMAGE, "condition": "res32"},
}

EXPERIMENTS = {
    "exp1": {"title": "Experiment 1: CIFAKE -> CIFAKE", "model_tag": "cifake", "test_dataset": CIFAKE},
    "exp2": {"title": "Experiment 2: GenImage 5 generators -> BigGAN + Midjourney", "model_tag": "genimage", "test_dataset": GENIMAGE},
    "exp3": {"title": "Experiment 3: CIFAKE model -> GenImage held-out test (no retraining)", "model_tag": "cifake", "test_dataset": GENIMAGE},
    "res32": {"title": "Additional: 32x32 resolution (GenImage, train+test at 32x32)", "model_tag": "res32", "test_dataset": GENIMAGE},
}
ROBUSTNESS_EXPERIMENTS = ["exp1", "exp2"]


def build_prediction_plan():
    """Every (experiment, condition) a model must be evaluated on."""
    plan = []
    for exp in ROBUSTNESS_EXPERIMENTS:
        for cond in ROBUSTNESS_CONDITIONS:
            plan.append({"experiment": exp, "model_tag": EXPERIMENTS[exp]["model_tag"],
                         "dataset": EXPERIMENTS[exp]["test_dataset"], "split": "test", "condition": cond})
    plan.append({"experiment": "exp3", "model_tag": "cifake", "dataset": GENIMAGE, "split": "test", "condition": "clean"})
    plan.append({"experiment": "res32", "model_tag": "res32", "dataset": GENIMAGE, "split": "test", "condition": "res32"})
    return plan


PREDICTION_PLAN = build_prediction_plan()


def build_feature_sets():
    """All (dataset, split, condition) feature files needed by SVM and RF."""
    sets = []
    for tag in MODEL_TAGS.values():
        for split in ("train", "val"):
            sets.append((tag["dataset"], split, tag["condition"]))
    for entry in PREDICTION_PLAN:
        sets.append((entry["dataset"], entry["split"], entry["condition"]))
    return list(dict.fromkeys(sets))  # unique, order preserved


FEATURE_SETS = build_feature_sets()

# ---------------------------------------------------------------------------
# SVM / Random Forest (selected on the VALIDATION split only)
# ---------------------------------------------------------------------------
SELECTION_METRIC = "roc_auc"  # validation ROC-AUC, ties broken by validation accuracy
SVM_PARAM_GRID = [
    {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    {"kernel": "linear", "C": 0.01},
    {"kernel": "linear", "C": 0.1},
]
SVM_CACHE_MB = 2000
RF_PARAM_GRID = [
    {"n_estimators": n, "max_depth": d}
    for n in (200, 500) for d in (None, 20)
]
RF_MAX_FEATURES = "sqrt"

# ---------------------------------------------------------------------------
# CNN (ResNet50 transfer learning)
# ---------------------------------------------------------------------------
CNN_DROPOUT = 0.3
CNN_OPTIMIZER = "Adam"
CNN_LOSS = "BCEWithLogitsLoss (= sigmoid + binary cross-entropy, numerically stable)"
CNN_STAGE1_LR = 1e-4
CNN_STAGE2_LR = 1e-5
CNN_STAGE1_MAX_EPOCHS = 5    # frozen backbone
CNN_STAGE2_MAX_EPOCHS = 15   # layer4 unfrozen  -> 20 epochs maximum in total
CNN_PATIENCE = 5             # early stopping on validation loss (per stage)
CNN_HFLIP = True             # only augmentation: random horizontal flip (train split only)
CNN_USE_AMP = True           # mixed precision on CUDA only
