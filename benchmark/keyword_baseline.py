"""
Keyword baseline in the style of ConText (Harkema et al., 2009).

Rule: a "someone else" trigger word (a relative or spouse noun, or a
family-history phrase) opens a scope that runs FORWARD until the end of the
clause: a period, a semicolon, or "but". If the target condition falls inside
any trigger's scope, predict OTHER; otherwise predict PATIENT.

This is a simplified ConText. It has no pseudo-triggers or backward scope, so
it is easy to explain and has known, predictable failure cases (e.g., the
informant structure: "Her mother reports that the patient has X").

Run from the repo root:
    python benchmark/keyword_baseline.py
Writes benchmark/results/keyword.jsonl
"""

import csv
import json
import re
from pathlib import Path

RELATIVES = [
    "mother", "father", "mom", "dad", "parent", "parents", "sister", "brother",
    "sibling", "siblings", "aunt", "uncle", "son", "daughter", "grandmother",
    "grandfather", "grandparent", "cousin", "half-brother", "half-sister",
    "husband", "wife", "partner", "spouse",
]
PHRASES = ["family history", "fhx", "fh"]
TRIGGER = re.compile(r"\b(" + "|".join(re.escape(t) for t in PHRASES + RELATIVES) + r")\b", re.I)
# a period ends the scope only if it ends a sentence, not a title like "Mr." or "Ms."
TERMINATOR = re.compile(r"(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bDr)\.(\s|$)|;|\bbut\b", re.I)


def predict(text, char_start):
    for m in TRIGGER.finditer(text):
        if m.end() > char_start:
            break                              # triggers after the target can't scope over it
        end = TERMINATOR.search(text, m.end())
        scope_end = end.start() if end else len(text)
        if m.end() <= char_start < scope_end:
            return "OTHER", m.group(0)
    return "PATIENT", ""


def main():
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader(open(root / "benchmark" / "test_set.csv", encoding="utf-8")))
    out = root / "benchmark" / "results" / "keyword.jsonl"
    out.parent.mkdir(exist_ok=True)
    correct = 0
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            pred, trig = predict(r["text"], int(r["char_start"]))
            correct += pred == r["label"]
            fh.write(json.dumps({"id": r["id"], "method": "keyword", "prediction": pred,
                                 "trigger": trig}) + "\n")
    print(f"wrote {len(rows)} predictions to {out} (overall accuracy {correct / len(rows):.1%})")


if __name__ == "__main__":
    main()
