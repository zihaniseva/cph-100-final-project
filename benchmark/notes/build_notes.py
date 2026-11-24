"""
Build the note-level dataset: synthetic multi-section clinical notes made from
the same 12 sentence structures as the sentence-level benchmark.

Each note mixes the patient's own conditions with conditions of relatives and a
spouse, and with negated mentions. Every condition appears at most once per
note, so each one has exactly one role:

    P    the patient has it                  -> belongs in the patient's list
    PN   the patient is said NOT to have it  -> does not belong
    O    someone else has it                 -> does not belong
    ON   someone else is said not to have it -> does not belong

Run from the repo root:
    python benchmark/notes/build_notes.py              # 150 notes
    python benchmark/notes/build_notes.py --n 20 --seed 1

Writes benchmark/notes/notes.jsonl, one note per line:
    note_id, text, patient (name used), mentions: [{canonical, surface, role,
    structure, section, sentence}]
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gen", root / "data" / "generate_dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

CONDITIONS = list(gen.CONDITIONS_TRAIN) + list(gen.CONDITIONS_HELDOUT)
FATAL, ABBREV = gen.FATAL, gen.ABBREV
RELATIVES = {**gen.FAMILY_TRAIN, **gen.FAMILY_HELDOUT}       # noun -> gender (or None)
SPOUSES, POSS, SUBJ = gen.SPOUSES, gen.POSS, gen.SUBJ
NAMES = {g: [r for r in refs if r[0].isupper()] for g, refs in gen.PATIENT_REFS.items()}

# (structure, template, roles for C1/C2, options)
# {P} patient (name or "the patient"), {PS} patient pronoun, {F}/{F2} "her mother",
# {FAM} bare relative noun, {RS} relative's pronoun, {S} "her husband", {AGE}
HPI = [
    ("narrative", "{P} presents with {C1}.", ("P",), {}),
    ("narrative", "{P} was diagnosed with {C1} last year.", ("P",), {}),
    ("informant", "{F} reports that {PS} has had {C1} for several weeks.", ("P",), {}),
    ("informant", "{P} reports that {F} was recently diagnosed with {C1}.", ("O",), {}),
    ("long_distance", "{P}, who was brought in today by {F}, has {C1}.", ("P",), {}),
    ("long_distance", "{F}, who was seen at several clinics over the past year, has {C1}.", ("O",), {}),
    ("pronoun_coreference", "{P} came in today with {F}. {RS} has {C1}.", ("O",), {"coref": True}),
    ("pronoun_coreference", "{P} came in today with {F}. {PS} has {C1}.", ("P",), {"coref": True}),
]
PMH = [
    ("section_fragment", "PMH: {C1}, {C2}.", ("P", "P"), {"abbrev": True}),
    ("fh_trigger_phrase", "{P} has a history of {C1}.", ("P",), {}),
    ("negation_scope", "{P} denies {C1}.", ("PN",), {}),
    ("multi_relative_negation", "{P} has {C1}, but {F} has no history of {C2}.", ("P", "ON"), {}),
    ("multi_relative_negation", "{F} has {C1}, but {P} has no history of {C2}.", ("O", "PN"), {}),
    ("contrast", "{P} has {C1}, and {F} has {C2}.", ("P", "O"), {}),
    ("contrast", "{F} has {C1}, and {P} has {C2}.", ("O", "P"), {}),
]
FHX = [
    ("section_fragment", "FHx: {C1} -- {FAM}.", ("O",), {"abbrev": True}),
    ("fh_trigger_phrase", "Family history of {C1} in {F}.", ("O",), {}),
    ("age_and_death", "{F} was diagnosed with {C1} at age {AGE}.", ("O",), {}),
    ("age_and_death", "{F} died at age {AGE} of complications of {C1}.", ("O",), {"fatal": True}),
    ("negation_scope", "No family history of {C1}.", ("ON",), {}),
    ("patient_subject_family_history", "{P} has a family history of {C1}.", ("O",), {}),
]
SOCIAL = [
    ("non_blood_relative", "{P} lives with {S}, who has {C1}.", ("O",), {}),
    ("non_blood_relative", "{S} reports that {PS} has {C1}.", ("P",), {}),
]
ASSESS = [
    ("section_fragment", "{C1}, stable.", ("P",), {"abbrev": True}),
    ("narrative", "{P} with {C1}, follow up in 3 months.", ("P",), {}),
]
SECTIONS = [("HPI", HPI, (2, 3)), ("PMH", PMH, (2, 3)), ("FAMILY HISTORY", FHX, (3, 4)),
            ("SOCIAL HISTORY", SOCIAL, (1, 1)), ("ASSESSMENT", ASSESS, (1, 1))]


def cap(s):
    return s[:1].upper() + s[1:]


def build_note(rng, note_id):
    g = rng.choice(["F", "M"])
    name = rng.choice(NAMES[g])
    used = set()

    def take(fatal=False):
        pool = [c for c in CONDITIONS if c not in used and (c in FATAL or not fatal)]
        c = rng.choice(pool)
        used.add(c)
        return c

    def relative(opposite=False):
        opts = [r for r, rg in RELATIVES.items() if not opposite or (rg and rg != g)]
        return rng.choice(opts)

    lines, mentions = [], []
    for section, templates, (lo, hi) in SECTIONS:
        k = rng.randint(lo, hi)
        chosen = rng.sample(templates, min(k, len(templates)))
        sentences = []
        for structure, tmpl, roles, opt in chosen:
            fam = relative(opposite=opt.get("coref", False))
            fam2 = rng.choice([r for r in RELATIVES if r != fam])
            spouse = rng.choice(SPOUSES[g])
            conds = [take(fatal=opt.get("fatal", False)) for _ in roles]
            surf = [ABBREV[c] if opt.get("abbrev") and ABBREV.get(c) and rng.random() < 0.6 else c
                    for c in conds]
            fill = dict(P=rng.choice([name, "the patient"]), PS=SUBJ[g], F=f"{POSS[g]} {fam}",
                        F2=f"{POSS[g]} {fam2}", FAM=fam, RS=SUBJ.get(RELATIVES[fam] or "F"),
                        S=f"{POSS[g]} {spouse}", AGE=rng.randint(40, 85),
                        C1=surf[0], C2=surf[1] if len(surf) > 1 else "")
            text = tmpl.format(**fill)
            text = ". ".join(cap(p) for p in text.split(". "))   # capitalize each sentence
            sentences.append(text)
            for c, s, role in zip(conds, surf, roles):
                mentions.append(dict(canonical=c, surface=s, role=role, structure=structure,
                                     section=section, sentence=text))
        if section in ("PMH", "ASSESSMENT") and any(t.startswith(("PMH:", )) for t in sentences):
            sentences = [t.replace("PMH: ", "") for t in sentences]
        lines.append(f"{section}: " + " ".join(sentences))
    return dict(note_id=note_id, patient=name, text="\n".join(lines), mentions=mentions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    notes = [build_note(rng, f"n{i:03d}") for i in range(args.n)]
    out = Path(__file__).with_name("notes.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for n in notes:
            fh.write(json.dumps(n) + "\n")

    from collections import Counter
    roles = Counter(m["role"] for n in notes for m in n["mentions"])
    per = [sum(m["role"] == "P" for m in n["mentions"]) for n in notes]
    print(f"wrote {len(notes)} notes to {out}")
    print(f"mentions by role: {dict(roles)}; patient conditions per note: {min(per)}-{max(per)}")
    print("mentions by structure:", dict(Counter(m["structure"] for n in notes for m in n["mentions"])))


if __name__ == "__main__":
    main()
