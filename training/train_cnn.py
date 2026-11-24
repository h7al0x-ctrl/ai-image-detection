"""
Two-stage CNN training.

Stage 1: frozen backbone, Adam lr=1e-4, up to 5 epochs
Stage 2: layer4 unfrozen, Adam lr=1e-5, up to 15 epochs (starts from best stage-1 weights)
Early stopping: patience 5 on validation loss (per stage).
Best checkpoint (lowest validation loss over both stages) -> checkpoints/cnn/cnn_<tag>.pt

Run:  python -m training.train_cnn --tag cifake      (Experiment 1 model, reused in Exp 3)
      python -m training.train_cnn --tag genimage    (Experiment 2 model)
      python -m training.train_cnn --tag res32       (32x32 experiment model)
"""
import argparse
import copy
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import config  # noqa: E402
from models.cnn_model import ResNet50Binary  # noqa: E402
from preprocessing.image_cache import make_loader  # noqa: E402
from utils import Timer, ensure_dirs, environment_info, get_device, log_run, set_seed  # noqa: E402


def checkpoint_path(tag):
    return config.CHECKPOINT_DIR / "cnn" / f"cnn_{tag}.pt"


def history_path(tag):
    return config.TABLE_DIR / f"cnn_{tag}_history.csv"


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(y)
        correct += ((torch.sigmoid(logits.float()) >= config.THRESHOLD).float() == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train(tag, force=False):
    path = checkpoint_path(tag)
    if path.exists() and not force:
        print(f"{path.name} exists - skipping (use --force to retrain)")
        return
    spec = config.MODEL_TAGS[tag]
    set_seed()
    device = get_device()
    use_amp = config.CNN_USE_AMP and device.type == "cuda"
    print(f"=== CNN / {tag}: {spec['dataset']} ({spec['condition']}), device={device}, AMP={use_amp} ===")

    train_loader, _ = make_loader(spec["dataset"], "train", spec["condition"], train=True)
    val_loader, _ = make_loader(spec["dataset"], "val", spec["condition"], train=False)
    model = ResNet50Binary(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history, best_state, best = [], None, {"val_loss": math.inf}
    epoch = 0
    stages = [(1, config.CNN_STAGE1_LR, config.CNN_STAGE1_MAX_EPOCHS),
              (2, config.CNN_STAGE2_LR, config.CNN_STAGE2_MAX_EPOCHS)]
    with Timer() as total_timer:
        for stage, lr, max_epochs in stages:
            if best_state is not None:
                model.load_state_dict(best_state)  # stage 2 starts from the best stage-1 weights
            model.set_stage(stage)
            optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
            stage_best, bad_epochs = math.inf, 0
            n_trainable = sum(p.numel() for p in model.trainable_parameters())
            print(f"\n--- Stage {stage}: lr={lr}, trainable params={n_trainable:,} ---")
            for _ in range(max_epochs):
                epoch += 1
                with Timer() as t:
                    tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
                    va_loss, va_acc = run_epoch(model, val_loader, device, criterion, use_amp=use_amp)
                history.append({"epoch": epoch, "stage": stage, "lr": lr, "train_loss": tr_loss,
                                "val_loss": va_loss, "train_acc": tr_acc, "val_acc": va_acc,
                                "seconds": round(t.seconds, 1)})
                improved = va_loss < best["val_loss"] - 1e-6
                print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                      f"val loss {va_loss:.4f} acc {va_acc:.4f} | {t.seconds:.0f}s {'*best*' if improved else ''}")
                if improved:
                    best = {"val_loss": va_loss, "val_acc": va_acc, "epoch": epoch, "stage": stage}
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model_state_dict": best_state, "tag": tag, **best,
                                "config": {"dropout": config.CNN_DROPOUT, "stage1_lr": config.CNN_STAGE1_LR,
                                           "stage2_lr": config.CNN_STAGE2_LR, "batch_size": config.BATCH_SIZE,
                                           "seed": config.SEED, "weights": config.PRETRAINED_WEIGHTS,
                                           "loss": config.CNN_LOSS, "hflip": config.CNN_HFLIP},
                                "history": copy.deepcopy(history), "environment": environment_info()}, path)
                if va_loss < stage_best - 1e-6:
                    stage_best, bad_epochs = va_loss, 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= config.CNN_PATIENCE:
                        print(f"early stopping in stage {stage} (no improvement for {bad_epochs} epochs)")
                        break

    hist = pd.DataFrame(history)
    hist.to_csv(history_path(tag), index=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["history"] = history
    ckpt["training_seconds"] = round(total_timer.seconds, 1)
    torch.save(ckpt, path)
    print(f"\nBEST: epoch {best['epoch']} (stage {best['stage']}) val loss {best['val_loss']:.4f} "
          f"val acc {best['val_acc']:.4f}; total time {total_timer.seconds / 60:.1f} min")
    log_run("student5_cnn", "train", tag=tag, best_epoch=best["epoch"], best_val_loss=best["val_loss"],
            best_val_acc=best["val_acc"], minutes=round(total_timer.seconds / 60, 1), device=str(device))
    plot_curves()


def plot_curves():
    """training_curves.png with loss and accuracy for every trained tag."""
    tags = [t for t in config.MODEL_TAGS if history_path(t).exists()]
    if not tags:
        return
    fig, axes = plt.subplots(len(tags), 2, figsize=(12, 3.6 * len(tags)), squeeze=False)
    for r, tag in enumerate(tags):
        h = pd.read_csv(history_path(tag))
        switch = h.loc[h["stage"] == 2, "epoch"].min()
        for c, (metric, name) in enumerate((("loss", "Loss"), ("acc", "Accuracy"))):
            ax = axes[r, c]
            ax.plot(h["epoch"], h[f"train_{metric}"], "o-", label="train")
            ax.plot(h["epoch"], h[f"val_{metric}"], "o-", label="validation")
            if pd.notna(switch):
                ax.axvline(switch - 0.5, color="gray", ls="--", label="start stage 2")
            ax.set_title(f"CNN ({tag}) - {name}")
            ax.set_xlabel("epoch")
            ax.legend()
    fig.tight_layout()
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURE_DIR / "training_curves.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", choices=list(config.MODEL_TAGS), nargs="+", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    for tag in args.tag:
        train(tag, args.force)


if __name__ == "__main__":
    main()
