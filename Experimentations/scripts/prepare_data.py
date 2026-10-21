"""Audit and split HyperKvasir UC images."""
import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['0-1', '1', '1-2', '2', '2-3', '3']
EXPECTED = {'0-1': 35, '1': 201, '1-2': 11, '2': 443, '2-3': 28, '3': 133}


def write_csv(path, rows, fields):
    # UTF-8 with BOM lets Excel display Chinese and paths correctly.
    
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT/'data/raw/hyper-kvasir-labeled-images.zip')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = ROOT/'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    rows, errors, groups = [], [], defaultdict(list)
    counts = Counter()
    with zipfile.ZipFile(args.archive) as archive:
        # Extract only UC images; never trust arbitrary paths inside an archive.
        
        members = [m for m in archive.infolist() if 'ulcerative-colitis' in m.filename and m.filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for member in sorted(members, key=lambda m: m.filename):
            parts = Path(member.filename).parts
            folder = next(p for p in parts if p.startswith('ulcerative-colitis'))
            grade = folder.replace('ulcerative-colitis-', '').replace('grade-', '')
            if grade not in LABELS:
                raise ValueError(f'Unknown label: {folder}')
            counts[grade] += 1
            raw = archive.read(member)
            sha = hashlib.sha256(raw).hexdigest()
            target = ROOT/'data/uc_original'/grade/Path(member.filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            try:
                with Image.open(target) as image:
                    image.verify()
                with Image.open(target) as image:
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
                    width, height = rgb.size
                    pixel_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                    # A perceptual hash flags similar frames for manual review, not automatic deletion.
                    
                    small = list(rgb.convert('L').resize((9, 8)).tobytes())
                    dhash = sum((small[y*9+x] > small[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
                row = dict(path=target.relative_to(ROOT).as_posix(), grade=grade, sha256=sha,
                           pixel_sha256=pixel_hash, width=width, height=height, dhash=f'{dhash:016x}')
                rows.append(row)
                groups[pixel_hash].append(row)
            except (OSError, ValueError) as exc:
                errors.append({'path': str(target), 'error': str(exc)})
    if dict(counts) != EXPECTED:
        raise ValueError(f'Unexpected archive counts: {dict(counts)}; expected {EXPECTED}')
    # Exclude conflicting duplicate labels; keep one copy of identical decoded pixels.
    
    clean, duplicates, conflicts = [], [], []
    for group in groups.values():
        if len({r['grade'] for r in group}) > 1:
            conflicts.extend(group)
        else:
            clean.append(group[0])
            duplicates.extend(group[1:])
    fields = ['path', 'grade', 'sha256', 'pixel_sha256', 'width', 'height', 'dhash']
    write_csv(output/'inventory.csv', rows, fields)
    write_csv(output/'duplicates.csv', duplicates, fields)
    write_csv(output/'conflicts.csv', conflicts, fields)
    similar = []
    for i, a in enumerate(clean):
        for b in clean[i+1:]:
            distance = (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count()
            if distance <= 4:
                similar.append(dict(path_a=a['path'], path_b=b['path'], distance=distance))
    write_csv(output/'similar_pairs_review.csv', similar, ['path_a', 'path_b', 'distance'])
    # Connected perceptual-similarity groups must remain in the same split.
    
    parent = {r['path']: r['path'] for r in clean}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for pair in similar:
        parent[find(pair['path_a'])] = find(pair['path_b'])
    from sklearn.model_selection import StratifiedGroupKFold
    for task, labels in [('three_class', ['1', '2', '3']), ('six_class', LABELS)]:
        selected = [dict(r, label=labels.index(r['grade'])) for r in clean if r['grade'] in labels]
        y = [r['label'] for r in selected]
        g = [find(r['path']) for r in selected]
        # Five fixed folds approximate 60/20/20 while keeping related images together.
        
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=args.seed)
        for fold, (_, indices) in enumerate(splitter.split(selected, y, g)):
            for idx in indices:
                selected[idx]['split'] = 'test' if fold == 0 else 'val' if fold == 1 else 'train'
                selected[idx]['group'] = g[idx]
        for split in ['train', 'val', 'test']:
            if {r['label'] for r in selected if r['split'] == split} != set(range(len(labels))):
                raise ValueError(f'{task}/{split} misses a class. Review grouping before experiments.')
        write_csv(output/f'{task}.csv', selected, fields+['label', 'split', 'group'])
    audit = dict(seed=args.seed, archive_counts=dict(counts), readable=len(rows),
                 retained=len(clean), duplicates=len(duplicates), conflicting=len(conflicts),
                 similar_pairs=len(similar), errors=errors,
                 limitation='Patient IDs unavailable; similarity grouping cannot ensure patient independence.')
    (output/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
