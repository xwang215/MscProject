
# File: train_vqa.py
# Purpose of this file: treat severity grading as Visual Question Answering (VQA). The
# ViLT model from Hugging Face reads an endoscopy image together with a written question
# ("what is the mayo endoscopic score ...?") and picks an answer ("1", "2" or "3").
# Three question sets can be used for training:
# main    - only the grading question (closest to plain classification);
# ordinal - adds yes/no questions "is the mayo score 2 or higher?" and "is the mayo score 3?";
# kvasir  - ordinal + ulcerative colitis images and questions from Kvasir-VQA
# (colour, location, text, box artefact, ...), which have no Mayo grade.
# Evaluation always asks only the grading question on the same test images as the classifiers.








# Example:  python scripts/train_vqa.py --mode ordinal

# On a computer with an NVIDIA GPU (e.g. Google Colab) the default --device auto uses the GPU.



# Block: imports
# Standard tools, PyTorch, the ViLT classes from Hugging Face and shared helpers.

import argparse
import copy
import csv
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms as T
from transformers import ViltConfig, ViltForQuestionAnswering, ViltProcessor

from common import ROOT, compute_metrics, load_images, load_manifest, pick_device, save_json, set_seed

# Block: the evaluated question
# The single grading question asked of every test image.

MAIN_Q = "what is the mayo endoscopic score of the ulcerative colitis?"


# Block: question builders
# Create helper questions from the grade, and read the optional Kvasir-VQA question table.

def ordinal_questions(grade):
    g = int(grade)
    return [("is the mayo score 2 or higher?", "yes" if g >= 2 else "no"),
            ("is the mayo score 3?", "yes" if g == 3 else "no")]


def load_kvasir(path):
    qa = defaultdict(list)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                qa[r["path"]].append((r["question"].lower(), r["answer"].lower()))
    return qa


# Block: sample builder
# Turn images into (image, question, answer) training triples for one epoch.

def build_samples(rows, mode, kvasir, rng, train, kvasir_per_epoch=300):
    samples = []
    if train and mode == "kvasir":
        for path, qa in rng.sample(sorted(kvasir.items()), min(kvasir_per_epoch, len(kvasir))):
            samples.append((path, *rng.choice(qa)))
    for r in rows:
        samples.append((r["path"], MAIN_Q, r["grade"]))
        if not train or mode == "main":
            continue
        extra = ordinal_questions(r["grade"])
        samples.append((r["path"], *rng.choice(extra)))
    return samples


# Block: model input and forward pass
# Convert triples into tensors (pixels + tokenised text) and run ViLT on them.

def encode(processor, cache, batch, tf, answer_to_id, device):
    images = torch.stack([tf(cache[p]) for p, _, _ in batch])
    text = processor.tokenizer([q for _, q, _ in batch], padding="max_length", truncation=True, max_length=40, return_tensors="pt")
    target = torch.tensor([answer_to_id[a] for _, _, a in batch])
    return images.to(device), text.to(device), target.to(device)


def forward(model, images, text):
    mask = torch.ones(images.shape[0], images.shape[2], images.shape[3], dtype=torch.long, device=images.device)
    return model(input_ids=text["input_ids"], attention_mask=text["attention_mask"], token_type_ids=text["token_type_ids"], pixel_values=images, pixel_mask=mask).logits


# Block: grade prediction
# Ask the grading question for each image and keep only the three grade answers.

def predict_grades(model, processor, cache, rows, tf, answer_to_id, grades, bs=16):
    model.eval()
    device = next(model.parameters()).device
    grade_ids = [answer_to_id[g] for g in grades]
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            batch = [(r["path"], MAIN_Q, r["grade"]) for r in rows[i:i + bs]]
            images, text, _ = encode(processor, cache, batch, tf, answer_to_id, device)
            logits = forward(model, images, text)[:, grade_ids]
            probs.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(probs)


def main():
    # Block: command-line options
    # Question mode and training hyper-parameters.
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dandelin/vilt-b32-finetuned-vqa")
    ap.add_argument("--mode", choices=["main", "ordinal", "kvasir"], default="kvasir")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--kvasir-per-epoch", type=int, default=300)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Block: data
    # Same 3-class images and split as the classifiers, plus Kvasir-VQA pairs in "kvasir" mode.
    
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    rows, grades = load_manifest("three")
    split = {s: [r for r in rows if r["split"] == s] for s in ["train", "val", "test"]}
    kvasir = load_kvasir(ROOT / "data" / "processed" / "kvasir_vqa_uc_qa.csv") if args.mode == "kvasir" else {}
    if args.mode == "kvasir" and not kvasir:
        print("Warning: kvasir_vqa_uc_qa.csv not found; mode 'kvasir' equals 'ordinal'.")

    # Block: answer vocabulary and model
    # Collect every possible answer, then load ViLT and give it a new answer layer of that size.
    
    answers = sorted(set(grades) | {"yes", "no"} | {a for qa in kvasir.values() for _, a in qa})
    answer_to_id = {a: i for i, a in enumerate(answers)}
    processor = ViltProcessor.from_pretrained(args.model)
    config = ViltConfig.from_pretrained(args.model, num_labels=len(answers), id2label=dict(enumerate(answers)), label2id=answer_to_id)
    model = ViltForQuestionAnswering.from_pretrained(args.model, config=config, ignore_mismatched_sizes=True)
    device = pick_device(args.device)
    model.to(device)

    # Block: image pipelines
    # Cache all images and define light training augmentation and a fixed evaluation resize.
    
    mean, std = processor.image_processor.image_mean, processor.image_processor.image_std
    cache = load_images(rows + [{"path": p} for p in kvasir], args.image_size + 32)
    tf_train = T.Compose([T.RandomResizedCrop(args.image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    tf_eval = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor(), T.Normalize(mean, std)])

    # Block: optimiser, schedule and loss
    # Same recipe as the classifiers: AdamW, warm-up + cosine, class-weighted label-smoothed cross-entropy.
    
    head = [p for n, p in model.named_parameters() if "classifier" in n]
    body = [p for n, p in model.named_parameters() if "classifier" not in n]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}], weight_decay=0.05)
    per_epoch = math.ceil(len(build_samples(split["train"], args.mode, kvasir, random.Random(0), True, args.kvasir_per_epoch)) / args.batch_size)
    total, warm = args.epochs * per_epoch, per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    counts = np.bincount([grades.index(r["grade"]) for r in split["train"]], minlength=3)
    w = torch.ones(len(answers))
    for g, c in zip(grades, counts):
        w[answer_to_id[g]] = counts.sum() / (3 * c)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.1)

    # Block: training loop with early stopping
    # Each epoch draws fresh questions, trains, then checks grading performance on the validation set.
    
    best, best_key, history, wait, start = None, (-1.0,), [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        samples = build_samples(split["train"], args.mode, kvasir, rng, True, args.kvasir_per_epoch)
        rng.shuffle(samples)
        run_loss = 0.0
        for i in range(0, len(samples), args.batch_size):
            images, text, target = encode(processor, cache, samples[i:i + args.batch_size], tf_train, answer_to_id, device)
            loss = loss_fn(forward(model, images, text), target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(target)
        p_val = predict_grades(model, processor, cache, split["val"], tf_eval, answer_to_id, grades)
        y_val = np.array([grades.index(r["grade"]) for r in split["val"]])
        m = compute_metrics(y_val, p_val.argmax(1), grades, n_boot=0)
        history.append({"epoch": epoch, "train_loss": round(run_loss / len(samples), 4), "val_acc": round(m["accuracy"], 4), "val_macro_f1": round(m["macro_f1"], 4), "val_qwk": round(m["qwk"], 4), "minutes": round((time.time() - start) / 60, 1)})
        print(history[-1], flush=True)
        if (m["macro_f1"],) > best_key:
            best_key, best, wait = (m["macro_f1"],), copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    # Block: test evaluation
    # Restore the best weights and ask the grading question once for every test image.
    
    model.load_state_dict(best)
    p_test = predict_grades(model, processor, cache, split["test"], tf_eval, answer_to_id, grades)
    y_test = np.array([grades.index(r["grade"]) for r in split["test"]])
    metrics = compute_metrics(y_test, p_test.argmax(1), grades)
    name = f"vilt_vqa_{args.mode}"
    metrics.update({"model": args.model, "name": name, "family": "VQA", "task": "three", "classes": grades, "mode": args.mode,
                    "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "epochs_run": len(history),
                    "train_minutes": round((time.time() - start) / 60, 1), "best_val_macro_f1": best_key[0], "answer_vocabulary": answers, "device": str(device),
                    "kvasir_vqa_images": len(kvasir), "kvasir_vqa_qa_pairs": sum(len(v) for v in kvasir.values()), "settings": vars(args)})

    # Block: save results
    # Write metrics, the learning curve and per-image answers to results/three/vilt_vqa_<mode>/.
    
    out = ROOT / "results" / "three" / name
    save_json(metrics, out / "metrics.json")
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        wr.writeheader()
        wr.writerows(history)
    with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "true", "pred"] + [f"p_{g}" for g in grades])
        for r, t, p in zip(split["test"], y_test, p_test):
            wr.writerow([r["path"], grades[t], grades[int(p.argmax())]] + [round(float(v), 4) for v in p])
    print({k: metrics[k] for k in ["accuracy", "macro_f1", "qwk", "mae_grade", "train_minutes"]}, flush=True)


# Block: script entry
# Run main() only when this file is executed directly.

if __name__ == "__main__":
    main()
