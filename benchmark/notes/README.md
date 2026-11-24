# Note-level evaluation

The sentence-level test is saturated: both Gemini models scored 100% on all 600 sentences. This second evaluation asks the same underlying question in a form closer to real use. The model gets a whole note and must **list the patient's own conditions**.

## The notes

`notes.jsonl` holds 150 synthetic notes with five sections: HPI, PMH, FAMILY HISTORY, SOCIAL HISTORY, ASSESSMENT. Each is assembled from the same 12 sentence structures. Every condition appears at most once in a note, and each mention is tagged with its role:

| Role | Meaning | Belongs in the patient's list? | Mentions |
| --- | --- | --- | --- |
| P | the patient has it | yes | 802 |
| PN | the patient is said not to have it | no | 98 |
| O | a relative or spouse has it | no | 797 |
| ON | a relative is said not to have it | no | 141 |

An example note:

```
HPI: Ms. Nguyen came in today with her son. She has hypothyroidism. Ms. Nguyen presents with depression.
PMH: Her half-brother has lung cancer, but the patient has no history of sleep apnea. Ms. Nguyen denies coronary artery disease. The patient has a history of arthritis.
FAMILY HISTORY: Family history of Parkinson's disease in her cousin. Her paternal uncle died at age 43 of complications of pancreatitis. Her son was diagnosed with scoliosis at age 47.
SOCIAL HISTORY: Ms. Nguyen lives with her husband, who has lupus.
ASSESSMENT: Breast cancer, stable.
```

Correct list: hypothyroidism, depression, arthritis, breast cancer.

## Run it

From the repo root, with `GEMINI_API_KEY` set:

```bash
python benchmark/notes/run_notes.py --model gemini-3.8-flash --limit 10      # pilot: check the replies first
python benchmark/notes/run_notes.py --model gemini-3.8-flash                 # all 150 notes
python benchmark/notes/run_notes.py --model gemini-3.1-flash-lite --tag lite
python benchmark/notes/analyze_notes.py
```

The keyword baseline has already been run (`results/keyword.jsonl`). Add `--dry-run` to test without an API key.

## What gets measured

- **Exact-match notes:** the listed conditions are exactly the patient's own, with nothing extra and nothing missing.
- **Patient recall:** the share of the patient's conditions that got listed.
- **Wrongly listed:** how often a relative's, spouse's, or negated condition ended up in the patient's list. This is the misattribution error that matters clinically.
- **By structure:** the same errors broken down by the sentence structure each condition came from. You can put this side by side with the sentence-level results, where every structure scored 100%.

## Check by hand

- **`results/tables/unmatched_<method>.csv`:** items the scorer couldn't match to any condition in the note. Each is either a synonym the matcher doesn't know (count it as correct and note it) or a condition the model made up. Judge each one yourself.
- **`results/tables/errors_<method>.csv`:** every error, with a blank column for your own error categories.

## Limits to state in the write-up

- **The keyword baseline gets help the LLM doesn't.** It's told where each condition is mentioned, while the LLM has to find them.
- **Matching is generous.** "Rheumatoid arthritis" counts as "arthritis," for example.
- **The notes are synthetic and templated.** Real notes are messier.
