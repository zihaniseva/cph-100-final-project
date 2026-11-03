# CPHC100 Final Project: Decoding Clinical Experiencer Information in a Small Language Model

Does a small language model internally track *whose* condition a clinical sentence describes, the patient's or a family member's? We probe every layer to find where that information is decodable and test whether it holds up when the wording changes.

- **[PROPOSAL.md](PROPOSAL.md)**: research question, hypotheses, methods, timeline
- **[data/](data/)**: synthetic dataset generator, the generated CSV, and a data card

## Quick start

```bash
cd data
python generate_dataset.py    # regenerates experiencer_dataset.csv (11,400 rows)
```

The generator uses only the Python standard library.
