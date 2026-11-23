"""
Score every results file against the test set and write tables for the
write-up. Plots are left to you: these tables are what you plot from.

Run from the repo root:
    python benchmark/analyze.py                 # real results only
    python benchmark/analyze.py --include-dry-run

Writes benchmark/results/tables/:
    by_structure.csv       accuracy per method x structure, with 95% Wilson CIs
    summary.csv            per method: overall, set A, set B, pair consistency, unparsed
    composition.csv        accuracy if the hard structures made up 0%, 10%, ..., 100% of a test set
    errors_<method>.csv    every wrong answer, for your error analysis
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
res_dir, tab_dir = here / "results", here / "results" / "tables"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def acc(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x):
    return "" if x != x else f"{100 * x:.1f}"   # x != x catches NaN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-dry-run", action="store_true")
    args = ap.parse_args()

    test = {r["id"]: r for r in csv.DictReader(open(here / "test_set.csv", encoding="utf-8"))}
    files = sorted(res_dir.glob("*.jsonl"))
    if not args.include_dry_run:
        files = [f for f in files if "DRYRUN" not in f.name]
    tab_dir.mkdir(parents=True, exist_ok=True)

    by_structure, summary, composition = [], [], []
    for f in files:
        method = f.stem
        preds = {}
        for line in open(f, encoding="utf-8"):
            rec = json.loads(line)
            preds[rec["id"]] = rec                    # last answer wins if an id repeats
        rows = [test[i] for i in preds]
        correct = {i: preds[i]["prediction"] == test[i]["label"] for i in preds}

        groups = defaultdict(list)
        for r in rows:
            groups[r["structure"]].append(correct[r["id"]])
        for s in sorted(groups):
            k, n = sum(groups[s]), len(groups[s])
            lo, hi = wilson(k, n)
            by_structure.append({"method": method, "structure": s,
                                 "test_set": next(r["test_set"] for r in rows if r["structure"] == s),
                                 "n": n, "accuracy_%": pct(k / n), "ci_low_%": pct(lo), "ci_high_%": pct(hi)})

        a = [correct[r["id"]] for r in rows if r["test_set"] == "A_typical"]
        b = [correct[r["id"]] for r in rows if r["test_set"] == "B_stress"]
        pairs = defaultdict(list)
        for r in rows:
            if r["pair_id"]:
                pairs[r["pair_id"]].append(correct[r["id"]])
        full_pairs = [v for v in pairs.values() if len(v) == 2]
        summary.append({
            "method": method, "n": len(rows),
            "overall_%": pct(acc(list(correct.values()))),
            "set_A_typical_%": pct(acc(a)), "set_B_stress_%": pct(acc(b)),
            "gap_A_minus_B_pts": pct(acc(a) - acc(b)) if a and b else "",
            "pairs_both_right_%": pct(acc([all(v) for v in full_pairs])),
            "pairs_one_right_%": pct(acc([sum(v) == 1 for v in full_pairs])),
            "n_pairs": len(full_pairs),
            "unparsed_or_error": sum(preds[i]["prediction"] not in ("PATIENT", "OTHER") for i in preds),
        })

        # test-set composition: mix set A and set B accuracy in different proportions
        if a and b:
            for share in range(0, 101, 10):
                composition.append({"method": method, "share_hard_%": share,
                                    "expected_accuracy_%": pct((1 - share / 100) * acc(a) + share / 100 * acc(b))})

        with open(tab_dir / f"errors_{method}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "structure", "text", "target", "label", "prediction", "raw", "your_error_category"])
            for r in sorted(rows, key=lambda r: r["structure"]):
                if not correct[r["id"]]:
                    p = preds[r["id"]]
                    w.writerow([r["id"], r["structure"], r["text"], r["condition_surface"], r["label"],
                                p["prediction"], (p.get("raw") or "").strip()[:200], ""])

    for name, data in (("by_structure", by_structure), ("summary", summary), ("composition", composition)):
        if data:
            with open(tab_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    print(f"scored {len(files)} result files -> {tab_dir}\n")
    for s in summary:
        print(f"{s['method']:28s} overall {s['overall_%']:>5}%  A {s['set_A_typical_%']:>5}%  "
              f"B {s['set_B_stress_%']:>5}%  pairs both right {s['pairs_both_right_%']:>5}%  "
              f"unparsed/error {s['unparsed_or_error']}")


if __name__ == "__main__":
    main()
