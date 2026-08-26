# MovieGPT data pipeline

MovieGPT is a reproducible cleaning and publishing pipeline for a single
English dialogue dataset that can be loaded directly for GPT-2 training. The
current private Hugging Face dataset combines:

- 236 complete *Friends* episode documents;
- 617 Cornell Movie-Dialogs movie documents; and
- 2,092,508 cleaned English OpenSubtitles v2024 chunks.

All 2,093,361 documents share one schema and one deterministic 90/5/5
train/validation/test split. The `source` column keeps provenance without
placing Friends or the other corpora in separate folders or configurations.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Downloaded source archives live under `data/raw/` and the generated Hugging
Face package lives under `dist/`; both are intentionally ignored by Git. The
checked-in Friends and Cornell outputs make the smaller cleaning results easy
to audit.

## Rebuild the data

The Friends and Cornell cleaners use their default ConvoKit input and output
locations:

```bash
.venv/bin/python scripts/clean_friends_corpus.py
.venv/bin/python scripts/clean_movie_corpus.py
```

Generate OpenSubtitles directly in the final package data directory so the
32 GiB plain-text input is never copied:

```bash
.venv/bin/python scripts/clean_opensubtitles.py \
  --output-dir dist/moviegpt/data \
  --workers 4
```

Then add the Friends and Cornell Parquet files, dataset card, and build
summary to that same package:

```bash
.venv/bin/python scripts/publish_moviegpt_dataset.py \
  --repo-id relentlessml/moviegpt \
  --opensubtitles-dir dist/moviegpt/data
```

To upload through the Hugging Face API after authenticating locally, append
`--upload`. Repositories are private by default; `--public` is an explicit
opt-in.

## Validate

```bash
.venv/bin/python scripts/clean_friends_corpus.py --check
.venv/bin/python scripts/clean_movie_corpus.py --check
.venv/bin/python scripts/clean_opensubtitles.py \
  --output-dir dist/moviegpt/data \
  --check
.venv/bin/python -m unittest discover -s tests
```

The complete build needs about 38 GiB for the retained input and output. Keep
45-50 GiB free when starting from scratch to allow comfortable operating
headroom. Once a Hub upload is verified, deleting the raw OpenSubtitles text
recovers about 32 GiB; the cleaned package is about 5.8 GiB.
