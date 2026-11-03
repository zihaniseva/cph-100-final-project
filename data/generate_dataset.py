"""
Synthetic dataset generator for the CPHC100 experiencer-probing project.

Each row is one (text, target condition) pair labeled PATIENT or OTHER, meaning
"who experiences the target condition". Sentences are organized into sentence
STRUCTURES drawn from how family history is actually written in clinical notes
and from the patterns known to break extraction systems:

  Wang et al. 2017   - free-text family history at Mayo is mostly semi-structured:
                       section headers, "condition -- relative" fragments,
                       relative-headed lists, abbreviations, age/death clauses.
  Shen et al. 2021   - n2c2/OHNLP 2019 schema: relative (+ maternal/paternal side,
                       half-blood), observation (+ negation), living status, age.
  Silva et al. 2020  - hard cases: several relatives in one sentence with one
                       negated, pronoun coreference across sentences, modifiers
                       before relatives, non-blood relatives (spouses).
  Harkema et al. 2009 (ConText) - experiencer triggers ("family history of",
                       relative nouns), scope, negation, informant-vs-experiencer.

Run:
  python generate_dataset.py                 -> experiencer_dataset.csv
  python generate_dataset.py --n 300 --seed 42

Every structure is balanced 50/50 by label. See data/README.md for the column
dictionary and the list of structures.
"""

import argparse
import csv
import random
from collections import Counter

# ----------------------------------------------------------------- conditions
# canonical name -> abbreviation used in note-style frames (None = no common one)
CONDITIONS_TRAIN = {
    "diabetes": "DM2", "hypertension": "HTN", "asthma": None, "migraines": None,
    "breast cancer": "breast CA", "depression": "MDD", "anemia": None,
    "epilepsy": None, "hypothyroidism": None, "osteoporosis": None,
    "colon cancer": "colon CA", "coronary artery disease": "CAD",
    "kidney stones": None, "glaucoma": None, "arthritis": None, "psoriasis": None,
    "anxiety": None, "gout": None, "high cholesterol": "HLD",
    "congestive heart failure": "CHF", "COPD": "COPD", "lupus": None,
    "celiac disease": None, "eczema": None, "sleep apnea": "OSA",
    "atrial fibrillation": "AFib", "Crohn's disease": None,
    "bipolar disorder": None, "lung cancer": "lung CA",
    "chronic kidney disease": "CKD",
}
CONDITIONS_HELDOUT = {
    "multiple sclerosis": "MS", "pancreatitis": None, "melanoma": None,
    "schizophrenia": None, "hepatitis C": "HCV", "Parkinson's disease": None,
    "scoliosis": None, "Huntington's disease": None, "tuberculosis": "TB",
    "a seizure disorder": None,
}
# conditions it is natural to say someone "died of complications of"
FATAL = {
    "diabetes", "breast cancer", "colon cancer", "coronary artery disease",
    "congestive heart failure", "COPD", "lung cancer", "chronic kidney disease",
    "pancreatitis", "melanoma", "hepatitis C", "Parkinson's disease", "Huntington's disease",
    "multiple sclerosis", "tuberculosis",
}
ABBREV = {**CONDITIONS_TRAIN, **CONDITIONS_HELDOUT}

# ----------------------------------------------------------------- relatives
# first- and second-degree blood relatives (n2c2 2019 definition), with gender
# for pronoun-coreference frames (None = gender-ambiguous term).
FAMILY_TRAIN = {
    "mother": "F", "father": "M", "sister": "F", "brother": "M",
    "aunt": "F", "uncle": "M", "son": "M", "daughter": "F",
}
FAMILY_HELDOUT = {  # includes Silva's modifier cases (maternal/paternal, great-, half-)
    "grandmother": "F", "grandfather": "M", "cousin": None,
    "maternal aunt": "F", "paternal uncle": "M", "half-brother": "M",
    "maternal grandmother": "F", "paternal grandfather": "M",
}
SPOUSES = {"F": ["husband", "partner"], "M": ["wife", "partner"]}  # non-blood

PATIENT_REFS = {
    "F": ["the patient", "the pt", "Ms. Garcia", "Ms. Chen", "Ms. Okafor", "Mrs. Lopez", "Ms. Nguyen"],
    "M": ["the patient", "the pt", "Mr. Patel", "Mr. Kim", "Mr. Novak", "Mr. Brown", "Mr. Haddad"],
}
POSS = {"F": "her", "M": "his"}
SUBJ = {"F": "she", "M": "he"}

# ----------------------------------------------------------------- structures
# Placeholders: {P} patient ref, {PPOS} patient possessive, {PSUBJ} patient
# subject pronoun, {F}/{F2} "her mother"-style relative phrases, {FAM} bare
# relative noun, {RSUBJ} relative's subject pronoun, {S} "her husband",
# {C} TARGET condition, {C2} second condition, {AGE}/{AGE2} ages.
# Frames in one structure that share `pair` are filled with identical words, so
# they form minimal pairs (same tokens, experiencer swapped).
# split: "train" structures are the lexically easy ones; "heldout" structures
# are the distractor / register-shift tests.
STRUCTURES = {
    "narrative": dict(
        split="train", source="baseline",
        desc="Plain narrative sentence; relative noun is a reliable cue.",
        frames=[
            ("PATIENT", "{P} has {C}.", None),
            ("PATIENT", "{P} was diagnosed with {C} last year.", None),
            ("OTHER", "{F} has {C}.", None),
            ("OTHER", "{F} was diagnosed with {C}.", None),
        ]),
    "fh_trigger_phrase": dict(
        split="train", source="ConText (Harkema 2009)",
        desc="ConText family-history trigger phrases vs. the patient's own history.",
        frames=[
            ("OTHER", "Family history of {C} in {F}.", None),
            ("OTHER", "Family history is significant for {C} in {F}.", None),
            ("PATIENT", "{P}'s past medical history is significant for {C}.", None),
            ("PATIENT", "{P} reports a past history of {C}.", None),
        ]),
    "section_fragment": dict(
        split="heldout", source="Wang 2017",
        desc="Semi-structured note text: section headers, 'condition -- relative' "
             "fragments, abbreviations; includes a section-header trap.",
        abbrev=True,
        frames=[
            ("OTHER", "FHx: {C} -- {FAM}.", None),
            ("OTHER", "FAMILY HISTORY: {FAM}: {C}, {C2}.", None),
            ("PATIENT", "PMH: {C}. FHx: {C2} -- {FAM}.", None),
            ("PATIENT", "FAMILY HISTORY: noncontributory. ASSESSMENT: {P} with {C}.", None),
        ]),
    "age_and_death": dict(
        split="heldout", source="Wang 2017; Shen 2021",
        desc="Age-at-onset and death clauses packed into one line.",
        fatal=True,
        frames=[
            ("OTHER", "{F} was diagnosed with {C} at age {AGE}.", "age"),
            ("PATIENT", "{P} was diagnosed with {C} at age {AGE}.", "age"),
            ("OTHER", "{F} died at age {AGE} of complications of {C}.", None),
            ("PATIENT", "{P} was diagnosed with {C} at age {AGE}; {F} died at {AGE2} of {C2}.", None),
        ]),
    "multi_relative_negation": dict(
        split="heldout", source="Silva 2020",
        desc="Two people and two conditions in one sentence, one of them negated; "
             "the target itself may be the negated one.",
        frames=[
            ("OTHER", "{F} had {C}, but {F2} has no history of {C2}.", None),
            ("OTHER", "{P} has {C2}, but {F} has no history of {C}.", None),
            ("PATIENT", "{P} has {C}, but {F} has no history of {C2}.", None),
            ("PATIENT", "{F} has {C2}, but {P} has no history of {C}.", None),
        ]),
    "pronoun_coreference": dict(
        split="heldout", source="Silva 2020",
        desc="Experiencer is only a pronoun in the second sentence; patient and "
             "relative have different genders, so the pronoun decides the label.",
        coref=True,
        frames=[
            ("OTHER", "{P} came in today with {F}. {RSUBJ} has {C}.", "coref"),
            ("PATIENT", "{P} came in today with {F}. {PSUBJ} has {C}.", "coref"),
        ]),
    "informant": dict(
        split="heldout", source="ConText (Harkema 2009)",
        desc="One person reports on another; the reporter is not the experiencer.",
        frames=[
            ("PATIENT", "{F} reports that {P} has {C}.", "rep"),
            ("OTHER", "{P} reports that {F} has {C}.", "rep"),
        ]),
    "patient_subject_family_history": dict(
        split="heldout", source="ConText (Harkema 2009)",
        desc="Patient is the grammatical subject, but 'family history' makes the "
             "family the experiencer.",
        frames=[
            ("PATIENT", "{P} has a history of {C}.", "pfh"),
            ("OTHER", "{P} has a family history of {C}.", "pfh"),
        ]),
    "contrast": dict(
        split="heldout", source="Silva 2020",
        desc="Patient and relative each have a different condition in one sentence.",
        frames=[
            ("PATIENT", "{P} has {C}, and {F} has {C2}.", "con"),
            ("OTHER", "{P} has {C2}, and {F} has {C}.", "con"),
        ]),
    "negation_scope": dict(
        split="heldout", source="ConText (Harkema 2009)",
        desc="Negated family history next to a positive finding; tests trigger scope.",
        frames=[
            ("PATIENT", "No family history of {C2}; {P} has {C}.", None),
            ("PATIENT", "{P} has {C}; family history is negative.", None),
            ("OTHER", "{P} denies {C2}, but {F} has {C}.", None),
            ("OTHER", "{P} denies any family history of {C}.", None),
        ]),
    "long_distance": dict(
        split="heldout", source="Silva 2020",
        desc="Relative and condition separated by a relative clause.",
        frames=[
            ("OTHER", "{F}, who was seen at several clinics over the past year, has {C}.", None),
            ("PATIENT", "{P}, who was brought in today by {F}, has {C}.", None),
        ]),
    "non_blood_relative": dict(
        split="heldout", source="Silva 2020; Shen 2021",
        desc="Spouse/partner as the other person; experiencer is OTHER but they are "
             "not family history in the hereditary sense.",
        spouse=True,
        frames=[
            ("OTHER", "{P}'s {SPOUSE} has {C}.", None),
            ("OTHER", "{P} lives with {S}, who has {C}.", None),
            ("PATIENT", "{P}, who lives with {S}, has {C}.", None),
            ("PATIENT", "{S} reports that {P} has {C}.", None),
        ]),
}

MARK = "\u0000TARGET\u0000"


def cap_first(s):
    return s[:1].upper() + s[1:]


def surface(cond, use_abbrev, rng):
    ab = ABBREV.get(cond)
    return ab if (use_abbrev and ab and rng.random() < 0.6) else cond


def sample_fillers(rng, spec, conds, fams):
    g = rng.choice(["F", "M"])
    pool = [c for c in conds if c in FATAL] if spec.get("fatal") else conds
    c, c2 = rng.sample(pool, 2)
    if spec.get("coref"):
        opp = "M" if g == "F" else "F"
        fam = rng.choice([f for f in fams if (FAMILY_TRAIN | FAMILY_HELDOUT)[f] == opp])
    else:
        fam = rng.choice(fams)
    fam2 = rng.choice([f for f in fams if f != fam])
    spouse = rng.choice(SPOUSES[g])
    rgender = (FAMILY_TRAIN | FAMILY_HELDOUT)[fam]
    ab = bool(spec.get("abbrev"))
    a1 = rng.randint(35, 70)
    return dict(
        gender=g, P=rng.choice(PATIENT_REFS[g]), PPOS=POSS[g], PSUBJ=SUBJ[g],
        FAM=fam, F=f"{POSS[g]} {fam}", F2=f"{POSS[g]} {fam2}",
        RSUBJ=SUBJ.get(rgender or "F"), SPOUSE=spouse, S=f"{POSS[g]} {spouse}",
        C=c, C2=c2, Csurf=surface(c, ab, rng), C2surf=surface(c2, ab, rng),
        AGE=a1, AGE2=rng.randint(a1 + 5, 89),
    )


def render(template, fill):
    """Fill a template; return (text, char_start, char_end) of the target."""
    keys = {k: v for k, v in fill.items()}
    keys["C"], keys["C2"] = MARK, fill["C2surf"]
    text = template.format(**keys)
    # capitalize the first letter of every sentence, keep everything else as is
    parts = text.split(". ")
    text = ". ".join(cap_first(p) for p in parts)
    start = text.index(MARK)
    target = fill["Csurf"]
    if start == 0:
        target = cap_first(target)
    text = text.replace(MARK, target)
    return text, start, start + len(target), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="examples per frame (or per minimal pair)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiencer_dataset.csv")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    conds = list(CONDITIONS_TRAIN) + list(CONDITIONS_HELDOUT)
    fams = list(FAMILY_TRAIN) + list(FAMILY_HELDOUT)

    rows, seen, pair_n = [], set(), 0
    for sname, spec in STRUCTURES.items():
        groups = {}
        for i, (label, tmpl, pair) in enumerate(spec["frames"]):
            groups.setdefault(pair or f"solo{i}", []).append((i, label, tmpl))
        for gkey, frames in groups.items():
            made = tries = 0
            while made < args.n and tries < args.n * 100:
                tries += 1
                fill = sample_fillers(rng, spec, conds, fams)
                batch = [(i, lab, t, *render(t, fill)) for i, lab, t in frames]
                if any(b[3] in seen for b in batch):
                    continue
                pid = ""
                if len(frames) > 1:
                    pair_n += 1
                    pid = f"mp{pair_n:05d}"
                for i, lab, tmpl, text, s, e, tgt in batch:
                    seen.add(text)
                    other_is_spouse = "{S}" in tmpl or "{SPOUSE}" in tmpl
                    other_is_family = any(k in tmpl for k in ("{F}", "{F2}", "{FAM}"))
                    used_conds = {fill["C"], fill["C2"]} if "{C2}" in tmpl else {fill["C"]}
                    used_fams = {fill["FAM"]} | ({fill["F2"].split(" ", 1)[1]} if "{F2}" in tmpl else set())
                    rows.append(dict(
                        text=text, condition=fill["C"], condition_surface=tgt,
                        char_start=s, char_end=e, label=lab,
                        structure=sname, frame_id=f"{sname}.{i}", source=spec["source"],
                        relation=("spouse" if other_is_spouse else
                                  "blood" if other_is_family else "none"),
                        family_term=fill["FAM"] if other_is_family else
                                    (fill["SPOUSE"] if other_is_spouse else ""),
                        patient_ref=fill["P"] if "{P}" in tmpl else "",
                        gender=fill["gender"],
                        target_negated=str("no history of {C}" in tmpl
                                           or "denies any family history of {C}" in tmpl).lower(),
                        abbreviated=str(tgt != fill["C"]).lower(),
                        pair_id=pid,
                        cond_split="heldout" if used_conds & set(CONDITIONS_HELDOUT) else "train",
                        family_split=("none" if not other_is_family else
                                      "heldout" if used_fams & set(FAMILY_HELDOUT) else "train"),
                        frame_split=spec["split"],
                    ))
                made += 1
            if made < args.n:
                print(f"warning: {sname}/{gkey} only produced {made} unique examples")

    # in-distribution split, grouped so minimal pairs never straddle splits
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["pair_id"] or f"solo{i}", []).append(r)
    keys = list(groups)
    rng.shuffle(keys)
    for j, k in enumerate(keys):
        split = "train" if j < 0.70 * len(keys) else "val" if j < 0.85 * len(keys) else "test"
        for r in groups[k]:
            r["random_split"] = split

    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = i

    cols = ["id", "text", "condition", "condition_surface", "char_start", "char_end",
            "label", "structure", "frame_id", "source", "relation", "family_term",
            "patient_ref", "gender", "target_negated", "abbreviated", "pair_id",
            "cond_split", "family_split", "frame_split", "random_split"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")
    print("labels:", dict(Counter(r["label"] for r in rows)))
    for (st, lb), c in sorted(Counter((r["structure"], r["label"]) for r in rows).items()):
        print(f"  {st:32s} {lb:8s} {c}")
    print("minimal pairs:", len({r["pair_id"] for r in rows if r["pair_id"]}))


if __name__ == "__main__":
    main()
