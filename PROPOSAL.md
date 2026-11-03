# Decoding Clinical Experiencer Information in a Small Language Model

**CPHC100 Final Project Proposal** · November 3, 2026 · *[team member names]*

## Summary

When a clinical note mentions a disease, someone has to decide whose disease it is. "The patient has diabetes" and "Her mother has diabetes" mention the same condition, but only one belongs in the patient's problem list. We will test whether a small open-weights language model internally represents this distinction, called the *experiencer* of a condition. We train simple linear probes on the model's hidden states at every layer, and test whether what the probes find holds up when the wording changes. Probing is the required deliverable. Activation patching, which tests whether the model actually *uses* the representation, is a stretch goal.

**Research question.** At which layers can we decode whether a disease belongs to the patient or a family member, and does that information generalize across wording?

## Background

**Clinical experiencer detection.** ConText (Harkema et al., 2009) defined experiencer as a property of a clinical condition: it belongs to the patient or to someone else. ConText detects it with trigger terms such as "mother" and "family history of." More recent work frames experiencer as one part of *assertion detection*, alongside certainty and temporality, and studies it with LLM in-context learning and LoRA fine-tuning (2024). That work evaluates model *outputs*, not internal representations.

**How family history is written.** Wang et al. (2017) analyzed free-text family history in Mayo Clinic notes. It is mostly semi-structured: section headers, short "condition -- relative" fragments, abbreviations, and age or death clauses. Silva et al. (2020) catalogued the sentences that break extraction systems. These include several relatives in one sentence with one negated, pronouns pointing back to an earlier relative, modifiers such as "maternal great-aunt," and spouses who aren't blood relatives.

**Probing.** A probe is a small classifier trained on a frozen model's hidden states to test whether some information is present there (Belinkov, 2022). Tenney, Das & Pavlick (2019) probed every layer of BERT and found an ordered progression: syntax becomes decodable earlier than semantic roles and coreference. High probe accuracy can be misleading, though, because a flexible probe can memorize its inputs. Hewitt & Liang (2019) address this with *control tasks*, random labelings that a memorizing probe also fits. They report *selectivity*: real-task accuracy minus control-task accuracy.

**Mechanism.** Feng & Steinhardt (2024) used causal interventions to show that language models bind entities to their attributes through specific internal representations. Linking a condition to its experiencer may rely on a similar binding mechanism. This motivates our stretch goal, but their work doesn't establish the clinical case.

**Our contribution.** Prior work studies clinical experiencer classification and the layer-wise accessibility of linguistic information. We investigate whether patient-versus-family attribution is linearly decodable across the layers of a small language model, and whether it generalizes beyond familiar lexical cues.

## Hypotheses

1. **H1, layer profile.** Experiencer is decodable above control baselines. Selectivity rises through early and middle layers and plateaus or dips near the output.
2. **H2, early-layer cue reliance.** Early-layer probes succeed mainly when a family word sits next to the condition. They fail on sentences like "Her mother reports that the patient has headaches," where a family word is present but the patient is the experiencer.
3. **H3, generalization.** Middle-layer probes transfer better than early-layer probes to unseen relative terms, unseen conditions, and unseen sentence structures.
4. **H4, mechanism (stretch).** Patching activations from a family-experiencer sentence into its patient-experiencer minimal pair shifts the model's answer at specific layers and positions.

## Data

**Synthetic dataset (primary).** The generator in `data/` produces 11,400 sentences across 12 sentence structures, labeled PATIENT or OTHER and balanced 50/50 within every structure. Two structures (plain narrative sentences and ConText trigger phrases) are the training structures, where a family word reliably signals OTHER. The other ten come from Wang et al., Shen et al., Silva et al., and ConText, and each breaks that shortcut differently: section fragments with abbreviations, age and death clauses, multiple relatives with negation, pronoun coreference, informants, "the patient has a family history of…", contrast, negation scope, long-distance dependencies, and spouses. The dataset includes 1,500 minimal pairs, which use the same words with the experiencer swapped. Full details are in [`data/README.md`](data/README.md).

Synthetic data lets us control exactly which cues appear. A keyword-based classifier scores about 50% on the minimal pairs by construction, so above-chance probe accuracy on them points to something other than keyword detection.

**Real-text check (secondary).** We will request access to the i2b2/n2c2 2010 assertion corpus, which labels problems "associated with someone else," through the n2c2 data use agreement this week. If access doesn't arrive by week 2, we will hand-write about 150 realistic note-style sentences, with two team members labeling them independently so we can report agreement. Real notes will be processed only on our own cloud VM, never through a hosted API.

## Model and compute

- **Primary model:** an open-weights instruction-tuned model of about 1.5B parameters (e.g., Qwen2.5-1.5B-Instruct). It is small enough to probe every layer cheaply and strong enough to answer "who has X?" questions, which patching requires.
- **Comparison model:** Pythia-1.4B, a base model widely used in interpretability research, for one replication of the main layer curve.
- **Compute:** the course's $2,000 Google Cloud credit, used through a GPU VM (L4) or Colab Enterprise. We expect the project to need well under $200. We'll set a billing alert and shut VMs down when idle.

## Methods

**1. Behavioral check.** Before probing, we test whether the model can do the task. For each sentence we ask "Who has [condition]? Answer: The" and compare the probability of "patient" with the relative's name. We report accuracy for each structure. Probing can still find information the model doesn't act on, but patching is limited to sentences the model gets right.

**2. Representation extraction.** We save the residual-stream hidden state after every layer, plus the embedding layer, at three token positions: the last token of the target condition (primary), the final token of the sentence, and the mean over the condition span.

**3. Probes.** One L2-regularized logistic regression per layer and position. We tune regularization on the validation split only. We report accuracy and macro-F1 with 95% confidence intervals across five seeds. The **best layer** is the one with the highest validation selectivity, and its test score is reported once.

**4. Baselines and controls.**

| Comparison | What it rules out |
| --- | --- |
| Majority class | Label imbalance |
| ConText-style keyword rule | That a simple trigger-word rule already solves the task |
| Bag-of-words logistic regression | That word presence alone explains probe accuracy |
| Layer 0 (embeddings only) | That context isn't needed |
| Randomly initialized model | That the architecture alone produces the signal |
| Control task: random label per condition word | Probe memorization (Hewitt & Liang); gives selectivity |
| Shuffled labels | Leakage; should be at chance |

**5. Generalization tests.**

| Test | Train on | Test on |
| --- | --- | --- |
| In-distribution | 70% of rows | 15% held-out test rows |
| Held-out relatives | Common relatives | e.g., "maternal grandmother," "half-brother," "cousin" |
| Held-out conditions | 30 conditions | 10 unseen conditions |
| Held-out structures | Narrative + trigger phrases | Each of the 10 distractor structures, reported separately |
| Minimal pairs | Any training split | Pairs with swapped experiencer |
| Real text | Synthetic data | n2c2 2010 or hand-written sentences |

A probe *generalizes* if its held-out selectivity stays close to its in-distribution selectivity.

**6. Stretch goal: activation patching.** For minimal pairs the model answers correctly, we run the corrupted sentence, overwrite the hidden state at one layer and position with the clean sentence's, and measure how much of the clean answer's logit difference is restored. Sweeping layers × positions gives a heatmap of where the experiencer information matters. If time allows, we will also steer the model by adding the probe's learned direction, and run a Feng & Steinhardt-style swap on contrast sentences. Patching starts only once the probing figures are done.

## Expected outputs and allowed claims

1. Layer curve: probe accuracy, control accuracy, and selectivity by layer (H1)
2. Accuracy by sentence structure at early, middle, and late layers, with baselines as reference lines (H2)
3. Generalization gap per held-out test, by layer (H3)
4. Minimal-pair accuracy by layer (H2, H3)
5. Patching heatmap (H4, stretch)

Plus an error analysis of 20–30 misclassified sentences at the best layer.

Probing shows information is *present* at a layer, not that the model uses it. We will claim use only from patching or steering results, and only for the sentence types tested. Results on synthetic templates aren't clinical validation; the real-text check shows how far they carry.

## Timeline

| Week | Dates | Work |
| --- | --- | --- |
| 1 | Nov 3–9 | Readings; request n2c2 access; finalize dataset; set up GCP; behavioral check |
| 2 | Nov 10–16 | Extract activations; layer-wise probes, baselines, control tasks; draft Figure 1 |
| 3 | Nov 17–23 | Generalization tests; minimal pairs; error analysis; Figures 1–4 final |
| 4 | Nov 24–Dec 1 | Patching (if Figures 1–4 are done); write-up and presentation |

**Gate:** if Figures 1–4 aren't finished by November 23, week 4 goes to completing and polishing the probing results instead of patching.

## Risks

| Risk | Fallback |
| --- | --- |
| Model fails the behavioral check on distractor structures | Probing still applies; patch only sentences it answers correctly, or switch models |
| Probes score near 100% at every layer, including layer 0 | Dataset is too easy; add harder structures and weight minimal pairs more |
| Control task also scores high | Use stronger regularization; rely on condition- and structure-based splits |
| n2c2 access doesn't arrive | Hand-written real-style sentences |
| Minimal pairs differ in token length | Keep only equal-length pairs for patching |

## References

- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*.
- Tenney, I., Das, D., & Pavlick, E. (2019). BERT Rediscovers the Classical NLP Pipeline. *ACL*.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP*.
- Harkema, H., et al. (2009). [ConText: An Algorithm for Determining Negation, Experiencer, and Temporal Status from Clinical Reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC2757457). *Journal of Biomedical Informatics*.
- Assertion Detection Large Language Model In-context Learning LoRA Fine-tuning (2024).
- Feng, J., & Steinhardt, J. (2024). How Do Language Models Bind Entities in Context? *ICLR*.
- Wang, Y., et al. (2017). [Systematic Analysis of Free-Text Family History in Electronic Health Record](https://pmc.ncbi.nlm.nih.gov/articles/PMC5543380).
- Shen, F., et al. (2021). [Family History Extraction From Synthetic Clinical Narratives Using NLP (n2c2/OHNLP 2019)](https://medinform.jmir.org/2021/1/e24008/). *JMIR Medical Informatics*.
- Silva, J. F., Almeida, J. R., & Matos, S. (2020). [Extraction of Family History Information From Clinical Notes: Deep Learning and Heuristics Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC7803476/). *JMIR Medical Informatics*.
