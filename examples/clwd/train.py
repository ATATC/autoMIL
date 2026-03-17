"""Autoresearch MIL: Editable experiment script for CLWD.

This is the ONLY file you modify. Everything in prepare.py is fixed.

--- MODIFICATION GUIDE ---

Things you CAN change (in this file):
  - TARGET: which task / encoder / model to optimize
  - CONFIG: training hyperparameters (LR, weight decay, dropout, etc.)
  - preprocess_features(): feature normalization, PCA, fusion
  - augment_batch(): patch dropout, feature noise, mixup
  - create_loss_fn(): focal loss, label smoothing, etc.
  - create_optimizer(): AdamW, SAM, different param groups
  - create_lr_schedule(): cosine, linear, warmup variations
  - The training loop itself (train_single_fold)

Things you CANNOT change:
  - prepare.py (read-only: data loading, evaluation metrics, splits)
  - The split assignments (same 5-fold CV as the benchmark)
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CLAM_DIR: set to your CLAM installation path
# CLAM_DIR = "/path/to/CLAM"
# if CLAM_DIR not in sys.path:
#     sys.path.insert(0, CLAM_DIR)

import prepare
from prepare import (
    ENCODER_DIMS,
    N_FOLDS,
    compute_metrics,
    create_fold_loaders,
    get_plan_path,
    print_results,
)

CLAM_MODELS = {"clam_sb", "clam_mb", "mil_fc"}


# =========================================================================
# TARGET: What to optimize. Change these to switch task/encoder/model.
# =========================================================================
TASK = "subtype"
ENCODER = "hoptimus1"
MODEL_TYPE = "clam_mb"
SEED = 42
GPU = int(os.environ.get("AUTORESEARCH_GPU", "0"))
EXPERIMENT_DESCRIPTION = os.environ.get("AUTORESEARCH_DESC", "baseline")


# =========================================================================
# CONFIG: Training hyperparameters. Tune freely.
# =========================================================================
CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.25,
    "hidden_dim": 512,
    "num_epochs": 100,
    "warmup_epochs": 5,
    "patience": 10,
    "batch_size": 32,
    "max_seq_length": 4096,
    # CLAM-specific (ignored for nnMIL models)
    "model_size": "small",
    "k_sample": 8,
    "bag_weight": 0.7,
    "instance_eval": True,
}


# =========================================================================
# PREPROCESSING
# =========================================================================
def preprocess_features(features: torch.Tensor) -> torch.Tensor:
    """Transform patch features before they enter the model."""
    return features


# =========================================================================
# AUGMENTATION
# =========================================================================
def augment_batch(
    features: torch.Tensor,
    bag_sizes: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Augment a training batch in-place."""
    return features, bag_sizes, labels


# =========================================================================
# LOSS FUNCTION
# =========================================================================
def create_loss_fn() -> nn.Module:
    """Create the loss function."""
    return nn.CrossEntropyLoss()


# =========================================================================
# OPTIMIZER
# =========================================================================
def create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )


# =========================================================================
# LR SCHEDULE
# =========================================================================
def create_lr_schedule(optimizer, total_steps: int):
    """Cosine schedule with linear warmup."""
    import math
    base_lr = CONFIG["learning_rate"]
    warmup_steps = (total_steps // CONFIG["num_epochs"]) * CONFIG["warmup_epochs"]

    def _update(step: int):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr

    return _update


# =========================================================================
# MODEL CREATION
# =========================================================================
def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    """Create a MIL model."""
    if model_type in CLAM_MODELS:
        from models.model_clam import CLAM_SB, CLAM_MB
        from models.model_mil import MIL_fc

        if model_type == "clam_sb":
            return CLAM_SB(
                gate=True, size_arg=CONFIG.get("model_size", "small"),
                dropout=dropout, k_sample=CONFIG.get("k_sample", 8),
                n_classes=num_classes, embed_dim=input_dim,
            )
        elif model_type == "clam_mb":
            return CLAM_MB(
                gate=True, size_arg=CONFIG.get("model_size", "small"),
                dropout=dropout, k_sample=CONFIG.get("k_sample", 8),
                n_classes=num_classes, embed_dim=input_dim,
            )
        else:
            return MIL_fc(
                size_arg=CONFIG.get("model_size", "small"),
                dropout=dropout, n_classes=num_classes, embed_dim=input_dim,
            )
    else:
        from nnMIL.network_architecture.model_factory import create_mil_model
        return create_mil_model(
            model_type=model_type, input_dim=input_dim,
            hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout,
        )


# =========================================================================
# FORWARD PASS
# =========================================================================
def forward_pass(
    model: nn.Module,
    model_type: str,
    features: torch.Tensor,
    coords: torch.Tensor,
    bag_sizes: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run forward pass, returning (logits, instance_loss)."""
    if model_type in CLAM_MODELS:
        batch_logits = []
        instance_losses = []

        for i in range(features.size(0)):
            n_real = bag_sizes[i].item()
            h = features[i, :n_real]

            if model_type == "mil_fc":
                output = model(h)
                batch_logits.append(output[0])
            else:
                use_inst = labels is not None and CONFIG.get("instance_eval", True)
                if use_inst:
                    output = model(h, label=labels[i], instance_eval=True)
                    batch_logits.append(output[0])
                    instance_losses.append(output[4]["instance_loss"])
                else:
                    output = model(h)
                    batch_logits.append(output[0])

        logits = torch.cat(batch_logits, dim=0)
        if instance_losses:
            avg_inst_loss = sum(instance_losses) / len(instance_losses)
            return logits, avg_inst_loss
        return logits, None

    # nnMIL models
    if model_type == "vision_transformer":
        max_len = features.size(1)
        mask = (
            torch.arange(max_len, device=bag_sizes.device)
            .unsqueeze(0).expand(features.size(0), -1)
            >= bag_sizes.unsqueeze(1)
        )
        output = model(features, coords=coords, mask=mask)
    else:
        output = model(features)

    if isinstance(output, dict):
        return output["logits"], None
    return output, None


# =========================================================================
# TRAINING LOOP
# =========================================================================
def train_single_fold(fold: int, device: torch.device) -> dict:
    """Train one fold. Returns {"val": metrics_dict, "test": metrics_dict}."""
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold)
        torch.cuda.empty_cache()

    plan_path = get_plan_path(TASK, ENCODER)
    input_dim = ENCODER_DIMS[ENCODER]
    num_classes = 7  # CLWD 7-class

    train_loader, val_loader, test_loader = create_fold_loaders(
        plan_path, fold=fold, batch_size=CONFIG["batch_size"],
        max_seq_length=CONFIG["max_seq_length"], seed=SEED + fold,
    )

    model = create_model(
        model_type=MODEL_TYPE, input_dim=input_dim,
        hidden_dim=CONFIG["hidden_dim"], num_classes=num_classes,
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [fold {fold}] params={n_params:,}, train={len(train_loader)} batches")

    optimizer = create_optimizer(model)
    loss_fn = create_loss_fn()
    total_steps = len(train_loader) * CONFIG["num_epochs"]
    lr_schedule = create_lr_schedule(optimizer, total_steps)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0
    global_step = 0

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features, coords, bag_sizes, labels = (
                batch[0].to(device), batch[1].to(device),
                batch[2].to(device), batch[3].to(device),
            )

            features = preprocess_features(features)
            features, bag_sizes, labels = augment_batch(features, bag_sizes, labels)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, instance_loss = forward_pass(
                    model, MODEL_TYPE, features, coords, bag_sizes, labels=labels,
                )
                bag_loss = loss_fn(logits, labels)

                if instance_loss is not None:
                    bag_weight = CONFIG.get("bag_weight", 0.7)
                    loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss
                else:
                    loss = bag_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_schedule(global_step)
            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

        val_metrics = _evaluate(model, val_loader, device)

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print(f"  [fold {fold}] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    val_metrics = _evaluate(model, val_loader, device)
    test_metrics = _evaluate(model, test_loader, device)

    print(f"  [fold {fold}] val_auc={val_metrics['auc_roc']:.4f} test_auc={test_metrics['auc_roc']:.4f}")

    del model, optimizer, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"val": val_metrics, "test": test_metrics}


def _evaluate(model, loader, device):
    """Evaluate model on a DataLoader."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                features, coords, bag_sizes, labels = batch[0], batch[1], batch[2], batch[3]
            else:
                features, coords, bag_sizes, labels = batch

            features = features.to(device)
            coords = coords.to(device)
            bag_sizes = bag_sizes.to(device)

            features = preprocess_features(features)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = forward_pass(model, MODEL_TYPE, features, coords, bag_sizes)

            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    return compute_metrics(all_labels, all_probs)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=== autoMIL: {TASK} | {ENCODER} | {MODEL_TYPE} ===")
    print(f"device={device}, gpu={GPU}, seed={SEED}")
    print(f"config={CONFIG}")

    t0 = time.time()
    val_results, test_results = [], []

    for fold in range(N_FOLDS):
        result = train_single_fold(fold, device)
        val_results.append(result["val"])
        test_results.append(result["test"])

    elapsed = time.time() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available() else 0.0
    )

    print_results(
        val_results, test_results, task=TASK,
        extra={
            "elapsed_seconds": f"{elapsed:.1f}",
            "peak_vram_mb": f"{peak_vram_mb:.1f}",
            "encoder": ENCODER,
            "model_type": MODEL_TYPE,
        },
    )

    # --- Auto-log to results.tsv ---
    import subprocess

    results_tsv = os.path.join(PROJECT_ROOT, "results.tsv")
    header = "commit\tval_auc\tval_bacc\ttest_auc\ttest_bacc\tcomposite\tdelta\tvram_gb\telapsed_min\tstatus\tdescription\n"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    val_auc = np.mean([m["auc_roc"] for m in val_results])
    val_bacc = np.mean([m["bacc"] for m in val_results])
    test_auc = np.mean([m["auc_roc"] for m in test_results])
    test_bacc = np.mean([m["bacc"] for m in test_results])
    vram_gb = peak_vram_mb / 1024
    elapsed_min = elapsed / 60

    if not os.path.exists(results_tsv):
        with open(results_tsv, "w") as f:
            f.write(header)

    composite = (test_auc + test_bacc) / 2
    best_prev_composite = 0.0
    with open(results_tsv, "r") as f:
        for line in f:
            if line.startswith("commit"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                try:
                    prev_comp = (float(parts[3]) + float(parts[4])) / 2
                    best_prev_composite = max(best_prev_composite, prev_comp)
                except ValueError:
                    pass
    status = "keep" if composite > best_prev_composite + 1e-6 else "discard"
    delta_composite = composite - best_prev_composite

    description = EXPERIMENT_DESCRIPTION
    row = (
        f"{commit}\t{val_auc:.6f}\t{val_bacc:.6f}\t{test_auc:.6f}\t{test_bacc:.6f}"
        f"\t{composite:.6f}\t{delta_composite:+.6f}\t{vram_gb:.1f}\t{elapsed_min:.1f}\t"
        f"{status}\t{description}\n"
    )
    with open(results_tsv, "a") as f:
        f.write(row)

    print(f"\nLogged to {results_tsv}")
