#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

p = Path("data/lcquad_test.json")
if not p.exists():
    print("data/lcquad_test.json not found")
    raise SystemExit(1)

bak = p.with_suffix('.json.bak')
shutil.copyfile(p, bak)
print(f"Backup saved to {bak}")

with open(p, 'r', encoding='utf-8') as f:
    all_q = json.load(f)

changed = 0
for q in all_q:
    if q.get('question') is None:
        nn = q.get('NNQT_question')
        if nn:
            q['question'] = nn
            changed += 1

if changed:
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(all_q, f, ensure_ascii=False, indent=2)
    print(f"Backfilled {changed} entries where 'question' was null using 'NNQT_question'.")
else:
    print("No entries needed backfilling.")
