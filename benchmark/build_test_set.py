"""
Build the benchmark test set, the few-shot examples, and the hand-check sheet
from data/experiencer_dataset.csv.

Run from the repo root:
    python benchmark/build_test_set.py

Writes (all in benchmark/):
    test_set.csv          50 rows per structure x 12 structures = 600 rows,
                          25 PATIENT / 25 OTHER in every structure, minimal
                          pairs kept together. Column `test_set` = "A_typical"
                          for the two plain structures, "B_stress" for the rest.
    fewshot_examples.csv  4 labeled examples for the few-shot prompt, taken only
                          from the plain structures and never from the test set.
    handcheck_sheet.csv   100 test rows (8 or 9 per structure) for you to check
                          by hand. Fill in label_ok, natural, and note.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

SEED = 2026
PER_STRUCTURE = 50          # rows per structure in the test set (half each label)
HANDCHECK_N = 100
PLAIN = {"narrative", "fh_trigger_phrase"}   # test set A ("typical" sentences)

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(root / "data" / "experiencer_dataset.csv", encoding="utf-8")))
rng = random.Random(SEED)

by_structure = defaultdict(list)
for r in rows:
    by_structure[r["structure"]].append(r)

half = PER_STRUCTURE // 2
test = []
for structure in sorted(by_structure):
    group = by_structure[structure]
    pairs = defaultdict(list)
    solo = {"PATIENT": [], "OTHER": []}
    for r in group:
        if r["pair_id"]:
            pairs[r["pair_id"]].append(r)
        else:
            solo[r["label"]].append(r)
    # keep the structure's own mix of paired and unpaired rows
    n_pairs = round(half * sum(len(v) for v in pairs.values()) / len(group))
    chosen = []
    for pid in rng.sample(sorted(pairs), n_pairs):
        chosen.extend(pairs[pid])
    for label in ("PATIENT", "OTHER"):
        chosen.extend(rng.sample(solo[label], half - n_pairs))
    for r in chosen:
        r["test_set"] = "A_typical" if structure in PLAIN else "B_stress"
    test.extend(chosen)

test_ids = {r["id"] for r in test}

# few-shot examples: plain structures only, not in the test set, 2 per label,
# and no condition that appears in the test set's plain rows
pool = [r for r in rows if r["structure"] in PLAIN and r["id"] not in test_ids]
fewshot = []
for label in ("PATIENT", "OTHER"):
    for structure in sorted(PLAIN):
        cands = [r for r in pool if r["label"] == label and r["structure"] == structure]
        fewshot.append(rng.choice(cands))

cols = ["id", "test_set", "structure", "frame_id", "text", "condition", "condition_surface",
        "char_start", "char_end", "label", "pair_id", "target_negated", "relation", "source"]
rng.shuffle(test)
with open(root / "benchmark" / "test_set.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(test)

with open(root / "benchmark" / "fewshot_examples.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "structure", "text", "condition_surface", "label"],
                       extrasaction="ignore")
    w.writeheader()
    w.writerows(fewshot)

# hand-check sheet: spread across structures
by_s = defaultdict(list)
for r in test:
    by_s[r["structure"]].append(r)
structures = sorted(by_s)
quota = {s: HANDCHECK_N // len(structures) for s in structures}
for s in rng.sample(structures, HANDCHECK_N % len(structures)):
    quota[s] += 1
sheet = []
for s in structures:
    sheet.extend(rng.sample(by_s[s], quota[s]))
rng.shuffle(sheet)
with open(root / "benchmark" / "handcheck_sheet.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "structure", "text", "target", "label", "label_ok (Y/N)", "natural (Y/N)", "note"])
    for r in sheet:
        w.writerow([r["id"], r["structure"], r["text"], r["condition_surface"], r["label"], "", "", ""])

print(f"test_set.csv: {len(test)} rows "
      f"({sum(r['test_set'] == 'A_typical' for r in test)} in A, "
      f"{sum(r['test_set'] == 'B_stress' for r in test)} in B), "
      f"{len({r['pair_id'] for r in test if r['pair_id']})} minimal pairs")
print(f"fewshot_examples.csv: {len(fewshot)} rows")
print(f"handcheck_sheet.csv: {len(sheet)} rows")
