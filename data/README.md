# Experiencer dataset

`experiencer_dataset.csv` contains 11,400 synthetic clinical sentences. Each row pairs one sentence with one target condition and labels who experiences that condition: `PATIENT` or `OTHER`. The two labels are split 50/50 within every sentence structure.

To regenerate the file (deterministic for a given seed):

```bash
python generate_dataset.py --n 300 --seed 42
```

`--n` is the number of examples per frame, or per minimal pair.

## Sentence structures

The sentences are organized into 12 structures. Each structure reflects a way family history is actually written in clinical notes, or a pattern that published work found difficult for extraction systems.

| Structure | Split | Example | Label | Motivated by |
| --- | --- | --- | --- | --- |
| `narrative` | train | Her daughter was diagnosed with breast cancer. | OTHER | Baseline |
| `fh_trigger_phrase` | train | Family history of COPD in his uncle. | OTHER | ConText trigger phrases |
| `section_fragment` | held-out | PMH: epilepsy. FHx: HTN -- maternal aunt. | PATIENT | Wang 2017: headers, "condition -- relative", abbreviations |
| `age_and_death` | held-out | Her father died at age 71 of complications of congestive heart failure. | OTHER | Wang 2017; Shen 2021: age and death clauses |
| `multi_relative_negation` | held-out | Ms. Okafor has depression, but her cousin has no history of eczema. | OTHER (eczema) | Silva 2020: several relatives, one negated |
| `pronoun_coreference` | held-out | The pt came in today with his aunt. She has kidney stones. | OTHER | Silva 2020: pronoun points back to a relative |
| `informant` | held-out | Her mother reports that Ms. Chen has pancreatitis. | PATIENT | ConText: the reporter isn't the experiencer |
| `patient_subject_family_history` | held-out | Mr. Kim has a family history of lupus. | OTHER | ConText: patient is the subject, family is the experiencer |
| `contrast` | held-out | Mr. Haddad has hypertension, and his cousin has sleep apnea. | PATIENT (hypertension) | Silva 2020: two people, two conditions |
| `negation_scope` | held-out | No family history of anemia; Mr. Brown has glaucoma. | PATIENT | ConText: negation scope |
| `long_distance` | held-out | Mr. Patel, who was brought in today by his mother, has kidney stones. | PATIENT | Silva 2020: subject far from condition |
| `non_blood_relative` | held-out | Mr. Brown lives with his wife, who has migraines. | OTHER | Silva 2020; Shen 2021: spouses aren't blood relatives |

The **train** structures are the ones where a family word next to the condition reliably signals OTHER. The **held-out** structures break that shortcut, either because a family word is present but the patient is the experiencer, or because the cue is far away, abbreviated, or carried by a pronoun.

## Columns

| Column | Meaning |
| --- | --- |
| `id` | Row number |
| `text` | The sentence or note fragment |
| `condition` | Canonical condition name (e.g., `hypertension`) |
| `condition_surface` | How the condition appears in `text` (e.g., `HTN`) |
| `char_start`, `char_end` | Character span of the target in `text`; map this to token positions with your tokenizer's offset mapping |
| `label` | `PATIENT` or `OTHER` |
| `structure`, `frame_id` | Which structure and which template produced the row |
| `source` | Paper that motivated the structure |
| `relation` | `blood`, `spouse`, or `none` (no other person mentioned) |
| `family_term` | The relative or spouse noun, if any |
| `patient_ref` | How the patient is referred to, if at all |
| `gender` | Patient gender used for pronouns |
| `target_negated` | `true` if the target condition itself is negated ("has no history of …") |
| `abbreviated` | `true` if the target appears as an abbreviation |
| `pair_id` | Shared by minimal-pair rows: same words, swapped experiencer (1,500 pairs) |
| `cond_split` | `heldout` if the row uses one of the 10 held-out conditions |
| `family_split` | `heldout` if it uses a held-out relative term (e.g., `maternal grandmother`, `half-brother`); `none` if no blood relative appears |
| `frame_split` | `train` or `heldout` structure (the distractor test) |
| `random_split` | In-distribution 70/15/15 split; minimal pairs never straddle splits |

## How to use the splits

- **Main layer curve:** train on `random_split == train`, pick the layer on `val`, report on `test`.
- **Held-out conditions:** train on `cond_split == train`, test on `heldout`.
- **Held-out relative terms:** train on `family_split in (train, none)`, test on `heldout`.
- **Distractor test:** train on `frame_split == train`, test on each held-out structure separately.
- **Minimal pairs:** group by `pair_id`. A probe that only detects family words scores about 50% on these by construction.

## Known limitations

- The data is synthetic and template-based, so it's a controlled testbed rather than real clinical text. Pair it with a small real-text check (n2c2 2010 or hand-written sentences).
- Templates use full sentences more than real notes do. Wang et al. found real family history sections are mostly short fragments.
- Labels follow the templates and have not been independently annotated. Spot-check before reporting results.
- Sex-specific conditions were removed so no patient or relative is given a condition that doesn't fit their sex.

## Sources

- Wang Y, et al. [Systematic Analysis of Free-Text Family History in Electronic Health Record](https://pmc.ncbi.nlm.nih.gov/articles/PMC5543380). 2017.
- Shen F, et al. [Family History Extraction From Synthetic Clinical Narratives Using NLP (n2c2/OHNLP 2019)](https://medinform.jmir.org/2021/1/e24008/). JMIR Med Inform, 2021.
- Silva JF, Almeida JR, Matos S. [Extraction of Family History Information From Clinical Notes: Deep Learning and Heuristics Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC7803476/). JMIR Med Inform, 2020.
- Harkema H, et al. [ConText: An Algorithm for Determining Negation, Experiencer, and Temporal Status from Clinical Reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC2757457). J Biomed Inform, 2009.
