
# File: common.py
# Purpose of this file: shared building blocks used by every training script —
# reading the manifest, caching images, data augmentation, a PyTorch dataset,
# class weights against imbalance, and all evaluation metrics (with bootstrap CIs).




# Block: imports
# Libraries for files, numbers, deep learning, images and metrics.

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torchvision import transforms as T

# Block: constants
# Project folder, the class lists of both tasks, and the numeric Mayo value of each label.

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"three": ["1", "2", "3"], "six": ["0-1", "1", "1-2", "2", "2-3", "3"]}
GRADE_VALUE = {"0-1": 0.5, "1": 1.0, "1-2": 1.5, "2": 2.0, "2-3": 2.5, "3": 3.0}


# Block: reproducibility
# Fix all random generators so the same command gives the same result.

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Block: device choice
# Pick where the network runs: "auto" uses an NVIDIA GPU when one is present (e.g. on Google Colab),
# otherwise the CPU, so the same command works on the laptop and in the cloud.


def pick_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# Block: data loading
# Read the manifest written by prepare_dataset.py and decode images once into memory.

def load_manifest(task):
    classes = TASKS[task]
    with open(ROOT / "data" / "processed" / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["grade"] in classes]
    for r in rows:
        r["label"] = classes.index(r["grade"])
    return rows, classes


def load_images(rows, size):
    cache = {}
    for r in rows:
        with Image.open(ROOT / r["path"]) as im:
            cache[r["path"]] = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return cache


# Block: image pipelines
# Training images get light, medically sensible augmentation; evaluation images are only resized.

def build_transforms(image_size, mean, std):
    train = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    evaluate = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])
    return train, evaluate


# Block: PyTorch dataset
# Wraps the cached images so PyTorch's DataLoader can serve them in mini-batches.

class UCDataset(torch.utils.data.Dataset):
    def __init__(self, rows, cache, transform):
        self.rows, self.cache, self.transform = rows, cache, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(self.cache[r["path"]]), r["label"]


# Block: class weights
# Grade 2 dominates the data, so rarer grades receive larger weights in the loss.

def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# Block: evaluation metrics
# Accuracy-type metrics, ordinal metrics (QWK, error in Mayo points), per-class scores,
# the confusion matrix and bootstrap 95 % confidence intervals.

def compute_metrics(y_true, y_pred, classes, n_boot=1000, seed=0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(classes)))
    value = np.array([GRADE_VALUE[c] for c in classes])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"),
        "mae_grade": float(np.abs(value[y_true] - value[y_pred]).mean()),
        "within_one_class": float((np.abs(y_true - y_pred) <= 1).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)},
    }
    rng = np.random.default_rng(seed)
    boot = {"accuracy": [], "macro_f1": [], "qwk": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        boot["accuracy"].append(accuracy_score(y_true[idx], y_pred[idx]))
        boot["macro_f1"].append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
        boot["qwk"].append(cohen_kappa_score(y_true[idx], y_pred[idx], labels=labels, weights="quadratic"))
    out["ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boot.items()}
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


# Block: saving
# Write any result dictionary to disk as readable JSON.

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
