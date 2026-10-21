"""Verify reproducibility and split isolation."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for task in ['three_class', 'six_class']:
        manifest = ROOT/f'data/processed/{task}.csv'
        with manifest.open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        # Reject overlap in paths, exact pixels, or similarity groups across splits.
        
        for field in ['path', 'pixel_sha256', 'group']:
            memberships = defaultdict(set)
            for row in rows:
                memberships[row[field]].add(row['split'])
            assert all(len(s) == 1 for s in memberships.values()), field
        assert len({r['path'] for r in rows}) == len(rows)
        print(task, {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test']})
        for result_file in (ROOT/'results').glob(f'{task}_*.json'):
            result = json.loads(result_file.read_text())
            assert result['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
            assert sum(map(sum, result['confusion_matrix'])) == result['counts']['test']
            assert 0 <= result['accuracy'] <= 1
            assert 0 <= result['macro_f1'] <= 1
    print('All manifest and recorded-result checks passed.')


if __name__ == '__main__':
    main()
