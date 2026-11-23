# Benchmark: does LLM accuracy on experiencer attribution depend on the test set?

Everything here runs from the repo root. Only `run_gemini.py` needs an API key.

| Step | Command | Output |
| --- | --- | --- |
| 1. Build the test set (done) | `python benchmark/build_test_set.py` | `test_set.csv`, `fewshot_examples.csv`, `handcheck_sheet.csv` |
| 2. Hand-check 100 sentences (you) | open `handcheck_sheet.csv`, fill in the last three columns | your notes |
| 3. Keyword baseline (done) | `python benchmark/keyword_baseline.py` | `results/keyword.jsonl` |
| 4. Gemini pilot | `python benchmark/run_gemini.py --method zero_shot --model MODEL --limit 50` | `results/gemini_zero_shot.jsonl` |
| 5. Gemini full runs | same command without `--limit`, then again with `--method few_shot` | `results/gemini_*.jsonl` |
| 6. Stability check | `python benchmark/run_gemini.py --method zero_shot --model MODEL --limit 100 --tag repeat` | `results/gemini_zero_shot_repeat.jsonl` |
| 7. Score everything | `python benchmark/analyze.py` | `results/tables/*.csv` |

Before step 4: `pip install google-genai`, then set `GEMINI_API_KEY` in your terminal. Never put the key in a file in this repo. To check the pipeline without a key, add `--dry-run` (fake answers, no cost, written to separate `_DRYRUN` files).

## The test set

600 sentences: 50 from each of the 12 sentence structures in `data/`, 25 PATIENT and 25 OTHER in every structure, and 112 minimal pairs kept whole.

- **Set A ("typical")**: the 2 plain structures, 100 sentences.
- **Set B ("stress test")**: the other 10 structures, 500 sentences.

## Hand-check guide

For each row, `label_ok` is Y if the label names the person the target condition is mentioned **about**. A negated mention still counts: "her cousin has no history of eczema" is about the cousin, so it's OTHER. `natural` is Y if a clinician could plausibly write the sentence. Report how many rows failed either check, and drop or fix those.

## Keyword baseline, already scored

A ConText-style rule: a relative, spouse, or family-history word opens a scope until the end of the clause, and a target inside that scope is OTHER. It scores 100% on set A and 80% on set B. All of its errors come from four structures where it scores 50%: informant, long distance, pronoun coreference, and non-blood relative. The other eight structures don't trip it up at all, which is worth discussing in the write-up.

## Design choices to be ready to explain

- The task definition in `prompts.py`, especially how negated mentions are labeled
- Why the few-shot examples come only from plain structures (so they don't teach the hard cases directly)
- Why temperature is 0 and why every raw response is saved
- What minimal-pair consistency adds over plain accuracy
- What `composition.csv` shows: how a headline accuracy shifts as hard sentences make up more of a test set
