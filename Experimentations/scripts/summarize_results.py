
# File: summarize_results.py
# Purpose of this file: gather every results/<protocol>/<model>/metrics.json into summary
# CSV tables, and draw the comparison bar charts and confusion matrices used in the report.


# Run it after training:  python scripts/summarize_results.py



# Block: imports
# Tables, JSON, plotting (without a screen) and arrays.

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT

# Block: settings
# One colour per model family and the display order of families.

COLOURS = {"CNN": "#2a6f97", "ViT": "#c4553a", "VQA": "#6a8d3a"}
ORDER = {"CNN": 0, "ViT": 1, "VQA": 2}
COLS = ["name", "family", "model", "parameters_millions", "accuracy", "balanced_accuracy", "macro_f1", "qwk", "mae_grade", "within_one_class", "train_minutes", "epochs_run"]


# Block: summary table
# Write one row per model with rounded metrics and bootstrap confidence intervals.

def write_table(runs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS + ["acc_ci95", "macro_f1_ci95", "qwk_ci95"])
        for m in runs:
            ci = m.get("ci95", {})
            w.writerow([round(m[c], 4) if isinstance(m.get(c), float) else m.get(c, "") for c in COLS] + ["%.3f-%.3f" % tuple(ci[k]) if k in ci else "" for k in ["accuracy", "macro_f1", "qwk"]])


# Block: bar charts
# Three side-by-side charts (accuracy, macro-F1, QWK) with 95 % CI error bars.

def bar_chart(runs, path, title):
    names = [m["name"] for m in runs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, label in zip(axes, ["accuracy", "macro_f1", "qwk"], ["Accuracy", "Macro-F1", "Quadratic weighted kappa"]):
        vals = [m[key] for m in runs]
        err = np.array([[m[key] - m["ci95"][key][0], m["ci95"][key][1] - m[key]] for m in runs]).T
        ax.bar(names, vals, yerr=err, capsize=3, color=[COLOURS.get(m.get("family"), "grey") for m in runs])
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Block: confusion matrices
# One small heat-map per model showing which true grades were predicted as which.

def confusion_grid(runs, path):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.4))
    for ax, m in zip(np.atleast_1d(axes), runs):
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=8)
        ax.set_xticks(range(len(m["classes"])), m["classes"], fontsize=7)
        ax.set_yticks(range(len(m["classes"])), m["classes"], fontsize=7)
        ax.set_title(m["name"], fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    # Block: loop over protocols
    # Fine-tuning (3- and 6-class), the linear probe and the learning-rate alternatives are summarised separately.
    
    for folder in ["three", "six", "linear_probe_three", "lr_alternatives_three"]:
        files = sorted((ROOT / "results" / folder).glob("*/metrics.json"))
        if not files:
            continue
        runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        runs.sort(key=lambda m: (ORDER.get(m.get("family"), 3), -m["macro_f1"]))
        write_table(runs, ROOT / "results" / f"summary_{folder}.csv")
        bar_chart(runs, ROOT / "results" / f"comparison_{folder}.png", f"Test-set comparison: {folder} (blue = CNN, red = ViT, green = VQA)")
        confusion_grid(runs, ROOT / "results" / f"confusion_matrices_{folder}.png")
        print(open(ROOT / "results" / f"summary_{folder}.csv", encoding="utf-8").read())


# Block: script entry
# Run main() only when this file is executed directly.

if __name__ == "__main__":
    main()
