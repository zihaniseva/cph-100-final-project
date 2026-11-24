"""
Note-level evaluation: give a model a whole synthetic note and ask for the
patient's own conditions. Saves every raw response.

Run from the repo root (API key set as GEMINI_API_KEY, as for run_gemini.py):
    python benchmark/notes/run_notes.py --model gemini-3.8-flash --limit 10    # pilot
    python benchmark/notes/run_notes.py --model gemini-3.8-flash               # all 150 notes
    python benchmark/notes/run_notes.py --model gemini-3.1-flash-lite --tag lite
    python benchmark/notes/run_notes.py --keyword        # keyword baseline, no API
    python benchmark/notes/run_notes.py --model fake --dry-run                 # test, no API

Output: benchmark/notes/results/<name>.jsonl, one note per line. Resumable:
rerun the same command and finished notes are skipped.
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
from keyword_baseline import predict as keyword_experiencer  # noqa: E402

SYSTEM_INSTRUCTION = (
    "You read synthetic clinical notes. List every medical condition that the patient "
    "personally has or has had.\n"
    "Do not include conditions that belong to family members, a spouse, or anyone else.\n"
    "Do not include conditions the note says the patient does not have.\n"
    "Copy each condition's name as it is written in the note.\n"
    'Reply with only a JSON array of strings, for example ["asthma", "HTN"].'
)
NEGATION = re.compile(r"\b(no history of|no family history of|denies|no)\s+$", re.I)


def user_prompt(text):
    return "Note:\n" + text


def parse_list(raw):
    """Return a list of strings, or None if the reply isn't a JSON list."""
    if raw is None:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    m = re.search(r"\[.*\]", s, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x) for x in out] if isinstance(out, list) else None


def keyword_list(note):
    """Keyword rule per mention: include it if no someone-else trigger scopes over it and
    it isn't negated. Note: it is GIVEN where each condition is, which the LLM is not."""
    out = []
    for m in note["mentions"]:
        s = m["sentence"]
        i = s.find(m["surface"])
        if i < 0:
            i = s.lower().find(m["surface"].lower())
        exp, _ = keyword_experiencer(s, i)
        clause = re.split(r"[.;]|\bbut\b", s[:i])[-1]
        if exp == "PATIENT" and not NEGATION.search(clause):
            out.append(m["surface"])
    return out


class Gemini:
    def __init__(self, model):
        from google import genai
        from google.genai import types
        self.client, self.model = genai.Client(), model
        self.config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION,
                                                  temperature=0.0,
                                                  response_mime_type="application/json")

    def ask(self, note):
        r = self.client.models.generate_content(model=self.model, contents=user_prompt(note["text"]),
                                                config=self.config)
        return r.text, getattr(r, "model_version", None)


class Fake:
    """Dry-run stand-in: returns the true list, sometimes adding one wrong condition."""

    def __init__(self):
        self.rng = random.Random(0)

    def ask(self, note):
        truth = [m["surface"] for m in note["mentions"] if m["role"] == "P"]
        wrong = [m["surface"] for m in note["mentions"] if m["role"] != "P"]
        if wrong and self.rng.random() < 0.4:
            truth.append(self.rng.choice(wrong))
        return json.dumps(truth), "fake-model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="exact Gemini model name")
    ap.add_argument("--keyword", action="store_true", help="run the keyword baseline instead")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.keyword and not args.model:
        ap.error("give --model, or --keyword")

    notes = [json.loads(l) for l in open(here / "notes.jsonl", encoding="utf-8")]
    if args.limit:
        notes = notes[:args.limit]
    name = ("keyword" if args.keyword else "gemini" + (f"_{args.tag}" if args.tag else "")) \
        + ("_DRYRUN" if args.dry_run else "")
    out = here / "results" / f"{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["note_id"] for l in open(out, encoding="utf-8")
                if json.loads(l).get("predicted") is not None}
    todo = [n for n in notes if n["note_id"] not in done]
    print(f"{len(notes)} notes, {len(done)} already done, {len(todo)} to run -> {out}")
    if not todo:
        print("Nothing to do: every note already has an answer.")
        return

    model = None if args.keyword else (Fake() if args.dry_run else Gemini(args.model))
    bad = 0
    with open(out, "a", encoding="utf-8") as fh:
        for i, n in enumerate(todo, 1):
            raw = version = error = None
            if args.keyword:
                pred = keyword_list(n)
                raw = json.dumps(pred)
            else:
                for attempt in range(5):
                    try:
                        raw, version = model.ask(n)
                        break
                    except Exception as e:            # noqa: BLE001
                        error = f"{type(e).__name__}: {e}"
                        if getattr(e, "code", None) in (400, 401, 403, 404):
                            sys.exit(f"Stopping: {error[:300]}\nFix this, then rerun the same command.")
                        wait = 2 ** attempt * 5
                        print(f"  {n['note_id']}: {error[:120]} - retrying in {wait}s")
                        time.sleep(wait)
                pred = parse_list(raw)
            bad += pred is None
            fh.write(json.dumps({"note_id": n["note_id"], "method": name,
                                 "model": None if args.keyword else args.model,
                                 "model_version": version, "predicted": pred, "raw": raw,
                                 "error": error if raw is None else None,
                                 "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {bad} unparsed or failed")
            if model and not args.dry_run:
                time.sleep(args.sleep)
    if bad:
        print("Some replies could not be read as a list; rerun the same command to retry them.")


if __name__ == "__main__":
    main()
