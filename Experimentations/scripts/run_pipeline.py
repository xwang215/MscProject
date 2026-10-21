
# File: run_pipeline.py
# Purpose of this file: run the whole study in the right order with one command —
# data preparation, CNN and Vision Transformer fine-tuning, the linear-probe comparison,
# the VQA experiments and the final summary. It works the same way on Windows, macOS and Linux.


# Example:  python scripts/run_pipeline.py            (downloads weights from Hugging Face)
# python scripts/run_pipeline.py --models-dir D:/models   (uses local copies)


# The default --device auto uses an NVIDIA GPU when one is present (e.g. on Google Colab).



# Block: imports
# Tools to parse options, find files and start other Python scripts.

import argparse
import subprocess
import sys
from pathlib import Path

# Block: experiment plan
# Every model with its family and the learning rate used in this study.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = [
    ("microsoft/resnet-50", "resnet50", "CNN", "5e-4"),
    ("facebook/convnext-tiny-224", "convnext_tiny", "CNN", "1e-4"),
    ("facebook/deit-small-patch16-224", "deit_small", "ViT", "5e-5"),
    ("microsoft/swin-tiny-patch4-window7-224", "swin_tiny", "ViT", "5e-5"),
    ("facebook/dinov2-small", "dinov2_small", "ViT", "1e-5"),
    ("google/vit-base-patch16-224", "vit_base", "ViT", "3e-5"),
]


# Block: helper to run one step
# Print the command, run it with the same Python, and stop everything if it fails.

def run(*args):
    cmd = [sys.executable, str(HERE / args[0]), *args[1:]]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    # Block: options
    # Where to find weights and which stages to skip.
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="", help="Folder with org__model sub-folders; empty = download.")
    ap.add_argument("--skip", nargs="*", default=[], choices=["data", "finetune", "probe", "vqa"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    src = lambda mid: str(Path(args.models_dir) / mid.replace("/", "__")) if args.models_dir else mid

    # Block: data preparation
    # Extract and split HyperKvasir; prepare Kvasir-VQA only if its archive is present.
    
    if "data" not in args.skip:
        run("prepare_dataset.py")
        if list((ROOT / "data" / "raw").glob("kvasir-vqa-uc*.tar")):
            run("prepare_kvasir_vqa.py")

    # Block: fine-tuning CNNs and ViTs
    # Same split, epochs, augmentation and loss for every model; only the learning rate differs.
    
    if "finetune" not in args.skip:
        for mid, name, family, lr in MODELS:
            run("train_classifier.py", "--model", src(mid), "--name", name, "--family", family, "--lr", lr, "--device", args.device)

    # Block: linear probe
    # Frozen features + logistic regression for every model (fast second protocol).
    
    if "probe" not in args.skip:
        for mid, name, family, _ in MODELS:
            run("linear_probe.py", "--model", src(mid), "--name", name, "--family", family, "--device", args.device)

    # Block: VQA
    # ViLT with the three question sets.
    
    if "vqa" not in args.skip:
        for mode in ["main", "ordinal", "kvasir"]:
            run("train_vqa.py", "--model", src("dandelin/vilt-b32-finetuned-vqa"), "--mode", mode, "--device", args.device)

    # Block: summary
    # Build the tables, figures and error analysis used in the report.
    
    run("summarize_results.py")
    run("analyze_errors.py")


# Block: script entry
# Run main() only when this file is executed directly.

if __name__ == "__main__":
    main()
