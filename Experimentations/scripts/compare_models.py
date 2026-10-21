"""Frozen pretrained image encoders + supervised classifier."""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'.cache/torch'))
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoImageProcessor, AutoModel

MODELS = {'resnet18': None, 'deit_tiny': 'facebook/deit-tiny-patch16-224',
          'swin_tiny': 'microsoft/swin-tiny-patch4-window7-224'}
REVISIONS = {'deit_tiny': 'b3428f18dcc7b543470d07f14b4a4157815d1880',
             'swin_tiny': 'd00d478bfaf1f417d34a1186673ad4b4be264c51'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--task', choices=['three_class', 'six_class'], default='three_class')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    manifest = ROOT/f'data/processed/{args.task}.csv'
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()
    # Frozen encoders provide a feasible CPU baseline; this is not full fine-tuning.
    
    if args.model == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
        revision = str(weights)
    else:
        model_id = MODELS[args.model]
        # Pin the exact downloaded checkpoint revision for future reproduction.
        
        revision = REVISIONS[args.model]
        preprocess = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        options = {'add_pooling_layer': False} if args.model == 'deit_tiny' else {}
        model = AutoModel.from_pretrained(model_id, revision=revision, **options)
    model = model.to(device).eval()
    features = []
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            images = []
            for row in rows[offset:offset+args.batch_size]:
                with Image.open(ROOT/row['path']) as im:
                    images.append(ImageOps.exif_transpose(im).convert('RGB'))
            # Use each pretrained model's own resizing and normalization rules.
            
            if args.model == 'resnet18':
                x = torch.stack([preprocess(im) for im in images]).to(device)
                encoded = model(x)
            else:
                inputs = preprocess(images=images, return_tensors='pt').to(device)
                hidden = model(**inputs).last_hidden_state
                # Use pretrained token features, avoiding an untrained optional pooler.
                
                encoded = hidden[:, 0] if args.model == 'deit_tiny' else hidden.mean(dim=1)
            features.append(encoded.cpu().numpy())
            print(f'{args.model}: {min(offset+args.batch_size, len(rows))}/{len(rows)}', flush=True)
    x = np.concatenate(features)
    y = np.array([int(row['label']) for row in rows])
    split = np.array([row['split'] for row in rows])
    train, val, test = [split == s for s in ['train', 'val', 'test']]
    # Fit scaling on training data only and select regularization on validation data only.
    
    best, best_f1, selected_c = None, -1, None
    validation = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=args.seed))
        clf.fit(x[train], y[train])
        score = f1_score(y[val], clf.predict(x[val]), average='macro', zero_division=0)
        validation.append({'C': c, 'macro_f1': score})
        if score > best_f1:
            best, best_f1, selected_c = clf, score, c
    # Evaluate the locked test split once after selecting the classifier.
    
    pred = best.predict(x[test])
    labels = ['1', '2', '3'] if args.task == 'three_class' else ['0-1', '1', '1-2', '2', '2-3', '3']
    result = dict(model=args.model, model_id=MODELS[args.model] or 'torchvision/resnet18', revision=revision,
                  protocol='Frozen pretrained encoder + class-balanced logistic regression; no encoder fine-tuning',
                  task=args.task, manifest_sha256=manifest_hash, seed=args.seed, device=device,
                  selected_C=selected_c, validation=validation, validation_macro_f1=best_f1,
                  counts={s: int((split == s).sum()) for s in ['train', 'val', 'test']},
                  accuracy=accuracy_score(y[test], pred), balanced_accuracy=balanced_accuracy_score(y[test], pred),
                  macro_f1=f1_score(y[test], pred, average='macro', zero_division=0),
                  quadratic_weighted_kappa=cohen_kappa_score(y[test], pred, weights='quadratic', labels=list(range(len(labels)))),
                  ordinal_index_mae=mean_absolute_error(y[test], pred),
                  confusion_matrix=confusion_matrix(y[test], pred, labels=list(range(len(labels)))).tolist(),
                  classification_report=classification_report(y[test], pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0),
                  elapsed_seconds=time.time()-start)
    # Bootstrap image-level uncertainty; this cannot replace patient-level validation.
    
    rng = np.random.default_rng(args.seed)
    bootstrap = []
    for _ in range(1000):
        idx = rng.integers(0, len(pred), len(pred))
        bootstrap.append(f1_score(y[test][idx], pred[idx], labels=list(range(len(labels))), average='macro', zero_division=0))
    result['macro_f1_image_bootstrap_95ci'] = np.quantile(bootstrap, [.025, .975]).tolist()
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    (out/f'{args.task}_{args.model}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/f'{args.task}_{args.model}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_label', 'predicted_label'])
        for row, guess in zip([r for r in rows if r['split'] == 'test'], pred):
            writer.writerow([row['path'], row['grade'], labels[guess]])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
