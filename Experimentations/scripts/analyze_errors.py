
# File: analyze_errors.py
# Purpose of this file: look deeper than the headline scores. For every 3-class model it
# measures how many mistakes are only one grade away, how the model does on images whose
# near-duplicates carry a different grade (label noise), and it uses McNemar's exact test
# to check whether two models really differ on the same test images.



# Run it after summarize_results.py:  python scripts/analyze_errors.py



# Block: imports
# Tables, JSON, combinations of model pairs and the binomial distribution for McNemar's test.

import csv
import itertools
import json

from scipy.stats import binomtest

from common import ROOT, save_json


# Block: read helpers
# Load one model's test predictions, and find images that sit in a "conflicting" similarity group.

def read_predictions(folder):
    with open(folder / "test_predictions.csv", encoding="utf-8") as f:
        return {r["path"]: (r["true"], r["pred"]) for r in csv.DictReader(f)}


def conflicting_images():
    out = set()
    with open(ROOT / "data" / "processed" / "similar_pairs.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["grade_a"] != r["grade_b"]:
                out.update([r["a"], r["b"]])
    return out


def main():
    # Block: per-model error profile
    # For each model: accuracy, share of errors that are one grade off, and accuracy on noisy vs clean images.
    
    noisy = conflicting_images()
    folders = sorted(p.parent for p in (ROOT / "results" / "three").glob("*/test_predictions.csv"))
    preds = {f.name: read_predictions(f) for f in folders}
    profile = {}
    for name, p in preds.items():
        wrong = [(t, q) for t, q in p.values() if t != q]
        adjacent = sum(abs(int(t) - int(q)) == 1 for t, q in wrong)
        on_noisy = [t == q for path, (t, q) in p.items() if path in noisy]
        on_clean = [t == q for path, (t, q) in p.items() if path not in noisy]
        profile[name] = {"errors": len(wrong), "adjacent_error_share": round(adjacent / max(1, len(wrong)), 3),
                         "noisy_test_images": len(on_noisy), "accuracy_noisy": round(sum(on_noisy) / max(1, len(on_noisy)), 3),
                         "accuracy_clean": round(sum(on_clean) / max(1, len(on_clean)), 3)}

    # Block: McNemar's exact test between every pair of models
    # Only images where exactly one of the two models is right carry information; under "no difference"
    # each model should win about half of them, which a binomial test checks.
    
    mcnemar = []
    for a, b in itertools.combinations(sorted(preds), 2):
        common_paths = preds[a].keys() & preds[b].keys()
        a_only = sum(preds[a][x][0] == preds[a][x][1] and preds[b][x][0] != preds[b][x][1] for x in common_paths)
        b_only = sum(preds[b][x][0] == preds[b][x][1] and preds[a][x][0] != preds[a][x][1] for x in common_paths)
        pval = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
        mcnemar.append({"model_a": a, "model_b": b, "only_a_correct": a_only, "only_b_correct": b_only, "p_value": round(float(pval), 4)})

    # Block: save
    # Store everything in results/error_analysis.json and print a short view.
    
    save_json({"label_noise_images_total": len(noisy), "per_model": profile, "mcnemar": mcnemar}, ROOT / "results" / "error_analysis.json")
    print(json.dumps(profile, indent=2))
    for m in mcnemar:
        if m["p_value"] < 0.05:
            print("significant:", m)


# Block: script entry
# Run main() only when this file is executed directly.

if __name__ == "__main__":
    main()
