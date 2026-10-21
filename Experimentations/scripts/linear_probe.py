
# File: linear_probe.py
# Purpose of this file: a cheap second way to compare models. The pretrained network is
# frozen (not trained); it only turns each image into a feature vector, and a simple
# logistic regression learns the Mayo grade from those vectors. This shows how useful each
# model's "off-the-shelf" features are, separately from fine-tuning.



# Example:  python scripts/linear_probe.py --model google/vit-base-patch16-224 --name vit_base --family ViT

# On a computer with an NVIDIA GPU (e.g. Google Colab) the default --device auto uses the GPU.



# Block: imports
# Arrays, PyTorch, scikit-learn classifier tools, Hugging Face loaders and shared helpers.

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

from common import ROOT, build_transforms, compute_metrics, load_images, load_manifest, pick_device, save_json, set_seed


# Block: feature extraction
# Pass all images through the frozen backbone and keep one summary vector per image.

def features(model, cache, rows, tf):
    out = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            x = torch.stack([tf(cache[r["path"]]) for r in rows[i:i + 32]])
            o = model(pixel_values=x.to(device))
            if getattr(o, "pooler_output", None) is not None:
                f = o.pooler_output.flatten(1)
            else:
                f = o.last_hidden_state[:, 0]
            out.append(f.cpu().numpy())
    return np.concatenate(out)


def main():
    # Block: options and data
    # Read the options, the fixed split and the images; load the frozen backbone.
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--family", default="")
    ap.add_argument("--task", choices=["three", "six"], default="three")
    ap.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda.")
    args = ap.parse_args()
    set_seed(42)
    torch.set_num_threads(2)
    rows, classes = load_manifest(args.task)
    proc = AutoImageProcessor.from_pretrained(args.model)
    _, tf = build_transforms(224, proc.image_mean, proc.image_std)
    cache = load_images(rows, 224)
    model = AutoModel.from_pretrained(args.model).eval()
    model.to(pick_device(args.device))

    # Block: features and labels
    # Compute features once for every image and line them up with labels and split names.
    
    x = features(model, cache, rows, tf)
    y = np.array([r["label"] for r in rows])
    s = np.array([r["split"] for r in rows])

    # Block: choose regularisation on validation
    # Try four strengths, train on the training set, keep the one with the best validation macro-F1.
    
    best = None
    for c in [0.001, 0.01, 0.1, 1.0]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
        clf.fit(x[s == "train"], y[s == "train"])
        f1 = compute_metrics(y[s == "val"], clf.predict(x[s == "val"]), classes, n_boot=0)["macro_f1"]
        if best is None or f1 > best[0]:
            best = (f1, c, clf)

    # Block: test and save
    # Score the chosen classifier once on the test set and save the metrics.
    
    m = compute_metrics(y[s == "test"], best[2].predict(x[s == "test"]), classes)
    m.update({"name": args.name, "family": args.family, "model": args.model, "protocol": "linear_probe", "C": best[1], "best_val_macro_f1": best[0], "classes": classes, "feature_dim": int(x.shape[1])})
    save_json(m, ROOT / "results" / f"linear_probe_{args.task}" / args.name / "metrics.json")
    print(args.name, {k: round(m[k], 3) for k in ["accuracy", "macro_f1", "qwk"]})


# Block: script entry
# Run main() only when this file is executed directly.

if __name__ == "__main__":
    main()
