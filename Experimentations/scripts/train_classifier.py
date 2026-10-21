
# File: train_classifier.py
# Purpose of this file: fine-tune one pretrained image model from Hugging Face — either a
# CNN (ResNet-50, ConvNeXt) or a Vision Transformer (ViT, DeiT, Swin, DINOv2) — to predict the
# Mayo endoscopic grade, pick the best epoch on the validation set, and evaluate it once on
# the test set. Every model uses exactly the same images, split, losses and metrics.



# Example:  python scripts/train_classifier.py --model microsoft/resnet-50 --name resnet50 --family CNN

# On a computer with an NVIDIA GPU (e.g. Google Colab) the default --device auto uses the GPU.



# Block: imports
# Standard tools, PyTorch, Hugging Face loaders and our shared helpers.

import argparse
import copy
import csv
import math
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

from common import ROOT, UCDataset, build_transforms, class_weights, compute_metrics, load_images, load_manifest, pick_device, save_json, set_seed


# Block: evaluation helper
# Runs the model on a whole split without learning and returns class probabilities.

def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(pixel_values=x.to(device)).logits
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    # Block: command-line options
    # Which model to train and with which hyper-parameters.
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id or a local folder.")
    ap.add_argument("--name", required=True, help="Short name used for the results folder.")
    ap.add_argument("--family", default="", help="CNN or ViT, only used for reporting.")
    ap.add_argument("--task", choices=["three", "six"], default="three")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Block: data
    # Load the fixed split, the model's normalisation values, cached images and the three data loaders.
    
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    rows, classes = load_manifest(args.task)
    split = {s: [r for r in rows if r["split"] == s] for s in ["train", "val", "test"]}

    processor = AutoImageProcessor.from_pretrained(args.model)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    cache = load_images(rows, args.image_size + 32)
    t_train, t_eval = build_transforms(args.image_size, mean, std)
    loaders = {
        "train": torch.utils.data.DataLoader(UCDataset(split["train"], cache, t_train), batch_size=args.batch_size, shuffle=True),
        "val": torch.utils.data.DataLoader(UCDataset(split["val"], cache, t_eval), batch_size=32),
        "test": torch.utils.data.DataLoader(UCDataset(split["test"], cache, t_eval), batch_size=32),
    }

    # Block: model
    # Load the pretrained network and replace its 1,000-class ImageNet head with a new head for our grades.
    
    config = AutoConfig.from_pretrained(args.model, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id={c: i for i, c in enumerate(classes)})
    model = AutoModelForImageClassification.from_pretrained(args.model, config=config, ignore_mismatched_sizes=True)
    n_params = sum(p.numel() for p in model.parameters())
    device = pick_device(args.device)
    model.to(device)

    # Block: optimiser, schedule and loss
    # AdamW with a faster head, one warm-up epoch then cosine decay, and a class-weighted,
    # label-smoothed cross-entropy (labels are imbalanced and somewhat noisy).
    
    head = [p for n, p in model.named_parameters() if "classifier" in n]
    body = [p for n, p in model.named_parameters() if "classifier" not in n]
    optimiser = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * args.head_lr_mult}], weight_decay=args.weight_decay)
    steps = args.epochs * len(loaders["train"])
    warm = len(loaders["train"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    weights = class_weights([r["label"] for r in split["train"]], len(classes))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    # Block: training loop with early stopping
    # Train for up to N epochs; after each epoch, score the validation set and keep the best weights.
    
    best, best_key, history, wait = None, (-1.0, 0.0), [], 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(pixel_values=x).logits, y)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += loss.item() * len(y)
            seen += len(y)
        p_val, y_val = evaluate(model, loaders["val"])
        m_val = compute_metrics(y_val, p_val.argmax(1), classes, n_boot=0)
        val_loss = float(-np.log(p_val[np.arange(len(y_val)), y_val] + 1e-9).mean())
        history.append({"epoch": epoch, "train_loss": round(total / seen, 4), "val_loss": round(val_loss, 4), "val_acc": round(m_val["accuracy"], 4), "val_macro_f1": round(m_val["macro_f1"], 4), "val_qwk": round(m_val["qwk"], 4), "minutes": round((time.time() - start) / 60, 1)})
        print(history[-1], flush=True)
        key = (m_val["macro_f1"], -val_loss)
        if key > best_key:
            best_key, best, wait = key, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    # Block: final test evaluation
    # Restore the best validation weights and score the untouched test set exactly once.
    
    model.load_state_dict(best)
    p_test, y_test = evaluate(model, loaders["test"])
    pred = p_test.argmax(1)
    metrics = compute_metrics(y_test, pred, classes)
    metrics.update({"model": args.model, "name": args.name, "family": args.family, "task": args.task, "classes": classes,
                    "parameters_millions": round(n_params / 1e6, 1), "epochs_run": len(history), "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history]) + 1),
                    "train_minutes": round((time.time() - start) / 60, 1), "best_val_macro_f1": best_key[0], "device": str(device),
                    "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"]), "settings": vars(args)})

    # Block: save results
    # Write the metrics, the learning curve and every test prediction to results/<task>/<name>/.
    
    out = ROOT / "results" / args.task / args.name
    save_json(metrics, out / "metrics.json")
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred"] + [f"p_{c}" for c in classes])
        for r, t, pr, pp in zip(split["test"], y_test, pred, p_test):
            w.writerow([r["path"], classes[t], classes[pr]] + [round(float(v), 4) for v in pp])
    print({k: metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "mae_grade", "train_minutes"]}, flush=True)


# Block: script entry
# Run main() only when this file is executed directly.

if __name__ == "__main__":
    main()
