"""Shared helpers: seeding, device selection, directories, run logging."""
import csv
import json
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config


def set_seed(seed=config.SEED, deterministic=True):
    """Seed Python, NumPy and PyTorch.

    Limitation: some CUDA kernels are not bit-for-bit deterministic, so CNN
    results on GPU can differ very slightly between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """CUDA if available, otherwise CPU (no hard CUDA assumption)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_worker(worker_id):
    """DataLoader worker seeding (keeps random flips reproducible)."""
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed=config.SEED):
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def ensure_dirs():
    for d in config.OUTPUT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


def log_run(component, action, status="ok", **details):
    """Append one row to logs/run_log.csv."""
    ensure_dirs()
    path = config.RUN_LOG
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "component", "action", "status", "details"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), component, action, status,
                         json.dumps(details, default=str, ensure_ascii=False)])


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}
    for name in ["torch", "torchvision", "sklearn", "pandas", "PIL", "imagehash", "matplotlib"]:
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = "not installed"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        pass
    return info


class Timer:
    """Usage:  with Timer() as t: ...;  t.seconds"""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.start
        return False


if __name__ == "__main__":
    ensure_dirs()
    info = environment_info()
    for k, v in info.items():
        print(f"{k:12s}: {v}")
    save_json(info, config.LOG_DIR / "environment.json")
    log_run("setup", "environment_check", **info)
    print("\nOK - folders created and environment saved to logs/environment.json")
