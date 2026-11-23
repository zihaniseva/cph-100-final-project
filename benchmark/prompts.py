"""
Prompt text shared by the Gemini runner. Kept in one place so the write-up can
quote it exactly.

Task definition (follows ConText's "experiencer"): the label is the person the
condition is mentioned ABOUT. A negated mention still has an experiencer:
"her cousin has no history of eczema" is about the cousin, so it is OTHER.
"""

import csv
from pathlib import Path

SYSTEM_INSTRUCTION = (
    "You read short excerpts from clinical notes. For one target condition in the "
    "excerpt, decide whose condition it is mentioned about.\n"
    "Answer PATIENT if the excerpt describes the patient's own condition, including "
    "when it says the patient does not have it.\n"
    "Answer OTHER if it describes someone else's condition, such as a family member's "
    "or a spouse's, including when it says that person does not have it.\n"
    "Reply with exactly one word: PATIENT or OTHER."
)


def item_text(text, target):
    return f"Excerpt: {text}\nTarget condition: {target}\nAnswer:"


def load_fewshot(path=None):
    path = path or Path(__file__).with_name("fewshot_examples.csv")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def build_prompt(method, text, target, fewshot=None):
    """Return the user-turn text for one test item."""
    if method == "zero_shot":
        return item_text(text, target)
    if method == "few_shot":
        shots = "\n\n".join(item_text(r["text"], r["condition_surface"]) + " " + r["label"]
                            for r in (fewshot or load_fewshot()))
        return "Examples:\n\n" + shots + "\n\nNow this one:\n\n" + item_text(text, target)
    raise ValueError(f"unknown method {method!r}")


if __name__ == "__main__":
    # print both prompts for one example, to paste into the write-up appendix
    print("SYSTEM:\n" + SYSTEM_INSTRUCTION + "\n")
    for m in ("zero_shot", "few_shot"):
        print(f"--- {m} ---\n" + build_prompt(m, "Her mother reports that the patient has headaches.",
                                            "headaches") + "\n")
