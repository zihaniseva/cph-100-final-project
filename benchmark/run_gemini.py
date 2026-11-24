"""
Send every test sentence to Gemini and save each raw response.

Setup (once, on your own computer):
    pip install google-genai
    export GEMINI_API_KEY="your key"        # Windows PowerShell: $env:GEMINI_API_KEY="your key"
Never paste the key into this file or commit it.

Run from the repo root:
    python benchmark/run_gemini.py --method zero_shot --model MODEL --limit 50   # pilot
    python benchmark/run_gemini.py --method zero_shot --model MODEL              # full run
    python benchmark/run_gemini.py --method few_shot  --model MODEL
    python benchmark/run_gemini.py --method zero_shot --model MODEL --limit 100 --tag repeat
                                                                                 # stability check
Test the whole pipeline without an API key or any cost:
    python benchmark/run_gemini.py --method zero_shot --model fake --dry-run

MODEL is the exact model name from AI Studio (for example "gemini-3.8-flash").
Write the name and the `model_version` saved in the results into your write-up.

The run is resumable: rerun the same command and it skips sentences that
already have an answer, so a crash or a rate-limit stop loses nothing.
Output: benchmark/results/gemini_<method>[_<tag>].jsonl, one JSON object per line.
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import SYSTEM_INSTRUCTION, build_prompt, load_fewshot  # noqa: E402

ANSWER = re.compile(r"\b(PATIENT|OTHER)\b")


def parse(raw):
    """PATIENT / OTHER if the reply is clearly one of them, else UNPARSED.
    A bare one-word reply counts in any case ("other"); in a longer reply only the
    capitalized words count, so "The patient's mother" is not read as PATIENT."""
    raw = (raw or "").strip()
    word = raw.strip(" .:*\"'`").upper()
    if word in ("PATIENT", "OTHER"):
        return word
    found = set(ANSWER.findall(raw))
    return found.pop() if len(found) == 1 else "UNPARSED"


class FakeModel:
    """Stand-in for Gemini used by --dry-run: right 85% of the time."""

    def __init__(self, labels):
        self.labels, self.rng = labels, random.Random(0)

    def ask(self, item_id, prompt):
        truth = self.labels[item_id]
        wrong = "OTHER" if truth == "PATIENT" else "PATIENT"
        return (truth if self.rng.random() < 0.85 else wrong), "fake-model"


class Gemini:
    def __init__(self, model, no_thinking):
        from google import genai
        from google.genai import types
        self.client, self.model = genai.Client(), model    # reads GEMINI_API_KEY
        cfg = dict(system_instruction=SYSTEM_INSTRUCTION, temperature=0.0)
        if no_thinking:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        self.config = types.GenerateContentConfig(**cfg)

    def ask(self, item_id, prompt):
        resp = self.client.models.generate_content(model=self.model, contents=prompt,
                                                   config=self.config)
        return resp.text, getattr(resp, "model_version", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["zero_shot", "few_shot"], required=True)
    ap.add_argument("--model", required=True, help="exact Gemini model name from AI Studio")
    ap.add_argument("--limit", type=int, help="only the first N sentences (pilot / stability run)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. 'repeat'")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between calls (rate limit)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="turn off the model's hidden reasoning (only some models allow this)")
    ap.add_argument("--dry-run", action="store_true", help="use a fake model; no key, no cost")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    rows = list(csv.DictReader(open(here / "test_set.csv", encoding="utf-8")))
    rows.sort(key=lambda r: int(r["id"]))            # fixed order, so --limit is reproducible
    if args.limit:
        rows = rows[:args.limit]

    name = f"gemini_{args.method}" + (f"_{args.tag}" if args.tag else "") + ("_DRYRUN" if args.dry_run else "")
    out = here / "results" / f"{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("prediction") != "ERROR":
                done.add(rec["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} sentences, {len(done)} already done, {len(todo)} to run -> {out}")
    if not todo:
        print("Nothing to do: every sentence already has an answer.")
        return

    model = (FakeModel({r["id"]: r["label"] for r in rows}) if args.dry_run
             else Gemini(args.model, args.no_thinking))
    fewshot = load_fewshot()
    counts = {"PATIENT": 0, "OTHER": 0, "UNPARSED": 0, "ERROR": 0}

    with open(out, "a", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            prompt = build_prompt(args.method, r["text"], r["condition_surface"], fewshot)
            raw, version, error = None, None, None
            for attempt in range(5):                  # retry with backoff on errors / rate limits
                try:
                    raw, version = model.ask(r["id"], prompt)
                    break
                except Exception as e:                # noqa: BLE001
                    error = f"{type(e).__name__}: {e}"
                    if getattr(e, "code", None) in (400, 401, 403, 404):
                        # wrong model name, bad key, no permission: retrying won't help
                        sys.exit(f"Stopping: {error[:300]}\nFix this, then rerun the same command.")
                    wait = 2 ** attempt * 5
                    print(f"  id {r['id']}: {error[:120]} - retrying in {wait}s")
                    time.sleep(wait)
            pred = parse(raw) if raw is not None else "ERROR"
            counts[pred] += 1
            fh.write(json.dumps({
                "id": r["id"], "method": args.method, "model": args.model,
                "model_version": version, "prediction": pred, "raw": raw,
                "error": error if raw is None else None,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }) + "\n")
            fh.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} {counts}")
            if not args.dry_run:
                time.sleep(args.sleep)

    if counts["UNPARSED"] or counts["ERROR"]:
        print("Some answers were UNPARSED or ERROR. Look at their `raw` / `error` fields; "
              "rerun the same command to retry the ERROR ones.")


if __name__ == "__main__":
    main()
