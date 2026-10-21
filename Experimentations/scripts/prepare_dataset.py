
# File: prepare_dataset.py
# Purpose of this file: extract the 851 ulcerative colitis (UC) images from the
# HyperKvasir archive, check that every image is intact, find duplicate and
# near-duplicate frames, and create one fixed, leakage-free train/validation/test split.


# Run it with:  python scripts/prepare_dataset.py



# Block: imports
# Load the standard-library and third-party tools this file needs.

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

# Block: constants
# Fixed values shared by the whole script: folders, label names and archive locations.

ROOT = Path(__file__).resolve().parents[1]
GRADES = ["0-1", "1", "1-2", "2", "2-3", "3"]
DEFAULT_ZIPS = [ROOT / "data" / "raw" / "hyper-kvasir-ulcerative-colitis.zip", ROOT / "hyper-kvasir-ulcerative-colitis.zip"]


# Block: helper functions
# Small reusable functions: a byte fingerprint and a perceptual fingerprint.

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def difference_hash(image, size=8):
    grey = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


# Block: union-find grouping
# A classic "union-find" structure that merges similar images into connected groups,
# so that every member of a group can later be placed in the same split.

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    # Block: command-line options
    # Let the user change the archive path, the near-duplicate threshold and the random seed.
    
    parser = argparse.ArgumentParser(description="Extract, audit and split the HyperKvasir UC images.")
    parser.add_argument("--zip", type=Path, default=None, help="Path to hyper-kvasir-ulcerative-colitis.zip or the full labelled archive.")
    parser.add_argument("--hash-threshold", type=int, default=6, help="Max Hamming distance treated as near-duplicate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Block: locate the archive and output folders
    # Find the downloaded archive and prepare the folders that will receive the results.
    
    archive = args.zip or next((p for p in DEFAULT_ZIPS if p.exists()), None)
    if archive is None:
        raise SystemExit("Archive not found. Put hyper-kvasir-ulcerative-colitis.zip in data/raw/.")

    image_dir = ROOT / "data" / "hyper-kvasir-uc"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Block: extract and audit every UC image
    # Verify the archive's checksums, copy each UC image out unchanged, confirm it decodes,
    # and record its size, average colour and fingerprints.
    
    rows = []
    with zipfile.ZipFile(archive) as zf:
        broken = zf.testzip()
        if broken is not None:
            raise SystemExit(f"Corrupted member in archive: {broken}")
        for info in zf.infolist():
            name = info.filename
            if name.endswith("image-labels.csv") or name.endswith("license.txt"):
                (image_dir).mkdir(parents=True, exist_ok=True)
                (image_dir / Path(name).name).write_bytes(zf.read(info))
                continue
            if "ulcerative-colitis-grade-" not in name or not name.lower().endswith(".jpg"):
                continue
            folder = Path(name).parent.name
            grade = folder.replace("ulcerative-colitis-grade-", "")
            if grade not in GRADES:
                raise SystemExit(f"Unknown grade folder: {folder}")
            data = zf.read(info)
            target = image_dir / f"grade-{grade}" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            with Image.open(target) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.float32)
                rows.append({
                    "path": target.relative_to(ROOT).as_posix(),
                    "file": target.name,
                    "grade": grade,
                    "width": im.width,
                    "height": im.height,
                    "mean_r": round(float(arr[..., 0].mean()), 2),
                    "mean_g": round(float(arr[..., 1].mean()), 2),
                    "mean_b": round(float(arr[..., 2].mean()), 2),
                    "sha256": sha256_bytes(data),
                    "dhash": difference_hash(im),
                })

    counts = Counter(r["grade"] for r in rows)
    print("Images per grade:", dict(sorted(counts.items())))

    # Block: find exact and near duplicates
    # Compare every pair of images; pairs with identical bytes or very similar perceptual
    # hashes (e.g. neighbouring video frames) are joined into one group and logged for review.
    
    uf = UnionFind(len(rows))
    similar_pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            same_bytes = rows[i]["sha256"] == rows[j]["sha256"]
            distance = bin(rows[i]["dhash"] ^ rows[j]["dhash"]).count("1")
            if same_bytes or distance <= args.hash_threshold:
                uf.union(i, j)
                similar_pairs.append({"a": rows[i]["path"], "b": rows[j]["path"], "grade_a": rows[i]["grade"], "grade_b": rows[j]["grade"], "hamming": distance, "identical_bytes": same_bytes})
    for idx, r in enumerate(rows):
        r["group"] = uf.find(idx)

    # Block: stratified, group-aware split
    # Cut the data into five folds that keep the grade proportions and never split a group;
    # fold 0 becomes the test set, fold 1 the validation set and folds 2-4 the training set.
    
    labels = [GRADES.index(r["grade"]) for r in rows]
    groups = [r["group"] for r in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_of = np.zeros(len(rows), dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        fold_of[test_idx] = fold
    split_names = {0: "test", 1: "val"}
    for idx, r in enumerate(rows):
        r["split"] = split_names.get(int(fold_of[idx]), "train")

    # Block: write the manifest and the review list
    # Save one row per image (used by every training script) and the list of similar pairs.
    
    fields = ["path", "file", "grade", "split", "group", "width", "height", "mean_r", "mean_g", "mean_b", "sha256", "dhash"]
    with open(processed_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["path"]):
            writer.writerow({**r, "dhash": f"{r['dhash']:016x}"})
    with open(processed_dir / "similar_pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "b", "grade_a", "grade_b", "hamming", "identical_bytes"])
        writer.writeheader()
        writer.writerows(similar_pairs)

    # Block: audit summary
    # Store the key facts about the data (counts, sizes, duplicates, split balance) in one JSON file.
    
    summary = {"archive": str(archive.name), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "images": len(rows), "per_grade": dict(sorted(counts.items())),
               "similar_pairs": len(similar_pairs), "groups": len(set(groups)),
               "label_conflicts_in_pairs": sum(p["grade_a"] != p["grade_b"] for p in similar_pairs),
               "split_counts": {s: dict(sorted(Counter(r["grade"] for r in rows if r["split"] == s).items())) for s in ["train", "val", "test"]},
               "width_range": [min(r["width"] for r in rows), max(r["width"] for r in rows)],
               "height_range": [min(r["height"] for r in rows), max(r["height"] for r in rows)],
               "mean_red_by_grade": {g: round(float(np.mean([r["mean_r"] for r in rows if r["grade"] == g])), 2) for g in GRADES},
               "seed": args.seed, "hash_threshold": args.hash_threshold}
    (processed_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# Block: script entry
# Run main() only when this file is executed directly, not when it is imported.

if __name__ == "__main__":
    main()
