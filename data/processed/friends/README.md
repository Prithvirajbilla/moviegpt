# Friends episode corpus

This directory is generated from ConvoKit's `friends-corpus` by
`scripts/clean_friends_corpus.py`.

## Training representation

- One UTF-8 `.txt` file per episode.
- One spoken utterance per line.
- A blank line between scenes.
- `<|endoftext|>` at the end of every episode.
- Original casing and punctuation are preserved.
- Unicode and whitespace are normalized conservatively.
- `TRANSCRIPT_NOTE` rows and empty/nonverbal utterances are excluded.
- Spoken text hidden behind stage directions in `transcript_with_note` is
  recovered when the primary `text` field is empty.
- Parenthesized/bracketed stage directions, credit/commercial/fade markers,
  and orphaned transcript-parser colons are removed.
- Embedded speaker labels leaked by the source transcript parser are removed,
  and the affected text is split back into separate training lines.
- Speaker names, stage directions, emotion labels, captions, and entity
  annotations are not serialized into the training text.

`manifest.jsonl` records the source, retained, and excluded counts plus byte
counts and SHA-256 digests for every episode. `summary.json` contains
corpus-wide totals, source-file digests, and record/byte/hash metadata for all
generated JSONL artifacts. `text_repairs.jsonl` preserves an audit trail of
every automatic repair, while `excluded_utterances.jsonl` records every
omitted source row and its exclusion reason. Validation recomputes scene and
line counts from the episode documents and verifies every recorded digest.

Regenerate the corpus from the repository root with:

```bash
python3 scripts/clean_friends_corpus.py
```

Validate the checked-in artifacts without regenerating them with:

```bash
python3 scripts/clean_friends_corpus.py --check
```
