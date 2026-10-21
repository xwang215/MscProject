
# File: prepare_kvasir_vqa.py
# Purpose of this file: unpack the ulcerative colitis part of Kvasir-VQA (images + question/
# answer pairs), keep the questions that describe what is visible in a UC image, and check
# whether any Kvasir-VQA image is a copy of a HyperKvasir image (which would leak test data).
# The output table is used by train_vqa.py in "kvasir" mode.



# Input: data/raw/kvasir-vqa-uc*.tar containing images/<img_id>.jpg and kvasir_vqa_uc.json.



# Block: imports
# Tools for archives, tables, JSON, counting, arrays and images.

import csv
import json
import tarfile
from collections import Counter

import numpy as np
from PIL import Image

from common import ROOT

# Block: questions to keep
# Kvasir-VQA asks about 20 questions per image; for UC images most are always "none"
# (polyps, instruments ...), so only the informative ones are kept.

KEEP = {
    "is there text?", "is there a green/black box artefact?", "is this finding easy to detect?",
    "what color is the abnormality? if more than one separate with ;", "where in the image is the abnormality?",
    "how many findings are present?",
}


# Block: perceptual hash
# A 128-bit fingerprint that stays the same when an image is resized or re-compressed.

def dual_dhash(im):
    rgb = im.convert("RGB")
    lum = np.array([0.299, 0.587, 0.114])
    h = np.asarray(rgb.resize((9, 8), Image.Resampling.BOX), dtype=np.float32) @ lum
    v = np.asarray(rgb.resize((8, 9), Image.Resampling.BOX), dtype=np.float32) @ lum
    return np.concatenate([(h[:, 1:] > h[:, :-1]).ravel(), (v[1:] > v[:-1]).ravel()]).astype(np.uint8)


def main():
    # Block: safe extraction
    # Unpack every part of the archive, refusing any path that tries to escape the target folder.
    
    out_dir = ROOT / "data" / "kvasir-vqa-uc"
    for archive in sorted((ROOT / "data" / "raw").glob("kvasir-vqa-uc*.tar")):
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = member.name.replace("\\", "/")
                if member.isfile() and ".." not in name and not name.startswith("/"):
                    target = out_dir / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
    data = json.loads((out_dir / "kvasir_vqa_uc.json").read_text(encoding="utf-8"))

    # Block: overlap check with HyperKvasir
    # Compare every Kvasir-VQA image with every HyperKvasir image; a tiny hash distance means the
    # same picture appears in both datasets. Copies of validation/test images must not be used for
    # training, otherwise the test set would leak into training.
    
    
    images = sorted({r[0] for r in data["qa"]})
    present = [i for i in images if (out_dir / "images" / f"{i}.jpg").exists()]
    with open(ROOT / "data" / "processed" / "manifest.csv", encoding="utf-8") as f:
        hk_rows = list(csv.DictReader(f))
    hk = np.stack([dual_dhash(Image.open(ROOT / r["path"])) for r in hk_rows])
    kv = np.stack([dual_dhash(Image.open(out_dir / "images" / f"{i}.jpg")) for i in present])
    dist = (kv[:, None, :] != hk[None, :, :]).sum(-1)
    nearest, nearest_idx = dist.min(1), dist.argmin(1)
    copy_of = {i: hk_rows[j] for i, d, j in zip(present, nearest, nearest_idx) if d <= 10}
    blocked = {i for i, r in copy_of.items() if r["split"] in ("val", "test")}

    # Block: filter question/answer pairs
    # Keep informative questions for downloaded, non-blocked images, and save them as a CSV.
    
    allowed = set(present) - blocked
    rows = []
    for img_id, q, a in data["qa"]:
        if q.lower() in KEEP and img_id in allowed:
            rows.append({"path": f"data/kvasir-vqa-uc/images/{img_id}.jpg", "img_id": img_id, "question": q.lower(), "answer": a.lower().strip()})
    with open(ROOT / "data" / "processed" / "kvasir_vqa_uc_qa.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "img_id", "question", "answer"])
        w.writeheader()
        w.writerows(rows)

    # Block: summary
    # Save counts, question/answer statistics, image sizes and the overlap result.
    
    summary = {"kvasir_vqa_uc_images": len(images), "downloaded": len(present), "qa_rows_kept": len(rows),
               "questions": Counter(r["question"] for r in rows), "top_answers": {q: Counter(r["answer"] for r in rows if r["question"] == q).most_common(6) for q in KEEP},
               "image_sizes": Counter(Image.open(out_dir / "images" / f"{i}.jpg").size for i in present[:200]).most_common(3),
               "near_copies_of_hyperkvasir": len(copy_of), "near_copies_by_split": Counter(r["split"] for r in copy_of.values()),
               "near_copy_grades": Counter(r["grade"] for r in copy_of.values()), "excluded_val_test_copies": len(blocked), "images_used": len(allowed)}
    summary["image_sizes"] = [[list(s), n] for s, n in summary["image_sizes"]]
    (ROOT / "data" / "processed" / "kvasir_vqa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# Block: script entry
# Run main() only when this file is executed directly.

if __name__ == "__main__":
    main()
