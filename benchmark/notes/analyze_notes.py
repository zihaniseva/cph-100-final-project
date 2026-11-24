"""
Score note-level results: match each listed condition to a mention in the
note, then count what was wrongly included and what was missed.

Run from the repo root:
    python benchmark/notes/analyze_notes.py
    python benchmark/notes/analyze_notes.py --include-dry-run

Writes benchmark/notes/results/tables/:
    summary.csv           per method: exact-match notes, patient recall, precision,
                          and how often each kind of non-patient condition got included
    by_structure.csv      per method x structure: miss rate for the patient's conditions,
                          wrong-inclusion rate for everyone else's
    errors_<method>.csv   every wrong inclusion and every miss, for your error analysis
    unmatched_<method>.csv  listed items that matched nothing in the note: check these
                          by hand (a missed synonym, or an invented condition)
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
tab = here / "results" / "tables"

# other names a model may use for a condition (lowercase)
SYNONYMS = {
    "diabetes": ["type 2 diabetes", "type 2 diabetes mellitus", "diabetes mellitus", "t2dm", "dm2", "dm"],
    "hypertension": ["htn", "high blood pressure"],
    "high cholesterol": ["hyperlipidemia", "hypercholesterolemia", "hld"],
    "coronary artery disease": ["cad"],
    "congestive heart failure": ["heart failure", "chf"],
    "sleep apnea": ["obstructive sleep apnea", "osa"],
    "atrial fibrillation": ["afib", "a-fib", "a fib"],
    "chronic kidney disease": ["ckd"],
    "depression": ["major depressive disorder", "mdd"],
    "breast cancer": ["breast ca"],
    "colon cancer": ["colon ca", "colorectal cancer"],
    "lung cancer": ["lung ca"],
    "multiple sclerosis": ["ms"],
    "hepatitis c": ["hcv", "hep c"],
    "tuberculosis": ["tb"],
    "migraines": ["migraine"],
    "kidney stones": ["kidney stone", "nephrolithiasis"],
    "a seizure disorder": ["seizure disorder", "seizures"],
    "copd": ["chronic obstructive pulmonary disease"],
}
ROLE_NAMES = {"P": "patient has", "PN": "patient negated", "O": "someone else has", "ON": "someone else negated"}


def norm(s):
    s = s.lower().strip().strip(".,;:\"'")
    s = re.sub(r"^(a|an|history of|h/o)\s+", "", s)
    return re.sub(r"\s+", " ", s)


def aliases(m):
    names = {m["canonical"], m["surface"]}
    names |= set(SYNONYMS.get(m["canonical"].lower(), []))
    return {norm(a) for a in names}


def match(item, mentions):
    """Index of the mention this listed item refers to, or None."""
    x = norm(item)
    best, best_len = None, 0
    for i, m in enumerate(mentions):
        for a in aliases(m):
            if x == a:
                return i
            if len(a) > 3 and len(x) > 3 and (a in x or x in a) and len(a) > best_len:
                best, best_len = i, len(a)
    return best


def pct(k, n):
    return "" if n == 0 else f"{100 * k / n:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-dry-run", action="store_true")
    args = ap.parse_args()
    notes = {n["note_id"]: n for n in map(json.loads, open(here / "notes.jsonl", encoding="utf-8"))}
    files = sorted((here / "results").glob("*.jsonl"))
    if not args.include_dry_run:
        files = [f for f in files if "DRYRUN" not in f.name]
    tab.mkdir(parents=True, exist_ok=True)

    summary, by_struct = [], []
    for f in files:
        method = f.stem
        recs = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            recs[r["note_id"]] = r
        role_n, role_in = defaultdict(int), defaultdict(int)
        s_n, s_err = defaultdict(int), defaultdict(int)
        exact = unparsed = n_listed = 0
        errors, unmatched = [], []
        for nid, r in recs.items():
            note = notes[nid]
            ms = note["mentions"]
            if r["predicted"] is None:
                unparsed += 1
                continue
            hit = set()
            for item in r["predicted"]:
                j = match(item, ms)
                if j is None:
                    unmatched.append([nid, item])
                else:
                    hit.add(j)
            n_listed += len(r["predicted"])
            truth = {i for i, m in enumerate(ms) if m["role"] == "P"}
            exact += hit == truth and not any(u[0] == nid for u in unmatched)
            for i, m in enumerate(ms):
                key = (m["structure"], "patient's own (missed)" if m["role"] == "P" else "someone else's / negated (wrongly listed)")
                role_n[m["role"]] += 1
                s_n[key] += 1
                included = i in hit
                role_in[m["role"]] += included
                wrong = (m["role"] == "P") != included
                s_err[key] += wrong
                if wrong:
                    errors.append([nid, "missed" if m["role"] == "P" else "wrongly listed",
                                   ROLE_NAMES[m["role"]], m["structure"], m["section"], m["surface"], m["sentence"], ""])
        scored = len(recs) - unparsed
        tp = role_in["P"]
        fp = sum(role_in[r] for r in ("PN", "O", "ON"))
        summary.append({
            "method": method, "notes_scored": scored, "unparsed": unparsed,
            "exact_match_notes_%": pct(exact, scored),
            "patient_recall_%": pct(tp, role_n["P"]),
            "precision_%": pct(tp, tp + fp + len(unmatched)),
            "someone_else_listed_%": pct(role_in["O"], role_n["O"]),
            "someone_else_negated_listed_%": pct(role_in["ON"], role_n["ON"]),
            "patient_negated_listed_%": pct(role_in["PN"], role_n["PN"]),
            "unmatched_items": len(unmatched),
        })
        for (s, kind), n in sorted(s_n.items()):
            by_struct.append({"method": method, "structure": s, "error_type": kind,
                              "n": n, "errors": s_err[(s, kind)], "error_rate_%": pct(s_err[(s, kind)], n)})
        with open(tab / f"errors_{method}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["note_id", "error", "true_role", "structure", "section", "condition", "sentence",
                        "your_error_category"])
            w.writerows(sorted(errors, key=lambda e: (e[3], e[0])))
        with open(tab / f"unmatched_{method}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["note_id", "listed_item", "your_judgment"])
            w.writerows(unmatched)

    for name, data in (("summary", summary), ("by_structure", by_struct)):
        if data:
            with open(tab / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    print(f"scored {len(files)} result files -> {tab}\n")
    for s in summary:
        print(f"{s['method']:18s} exact {s['exact_match_notes_%']:>5}%  recall {s['patient_recall_%']:>5}%  "
              f"precision {s['precision_%']:>5}%  other listed {s['someone_else_listed_%']:>5}%  "
              f"neg. listed {s['patient_negated_listed_%']:>5}%  unmatched {s['unmatched_items']}")


if __name__ == "__main__":
    main()
