#!/usr/bin/env python3
"""Build and optionally upload the GPT-2-ready MovieGPT Hub dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

if __package__:
    from .clean_friends_corpus import END_OF_TEXT, iter_jsonl, sha256_file
    from .clean_friends_corpus import validate_cleaned_corpus as validate_friends
    from .clean_movie_corpus import validate_cleaned_corpus as validate_movies
    from .clean_opensubtitles import validate_opensubtitles
    from .moviegpt_schema import DOCUMENT_FIELDS, SPLITS, document_split, parquet_schema
else:
    from clean_friends_corpus import END_OF_TEXT, iter_jsonl, sha256_file
    from clean_friends_corpus import validate_cleaned_corpus as validate_friends
    from clean_movie_corpus import validate_cleaned_corpus as validate_movies
    from clean_opensubtitles import validate_opensubtitles
    from moviegpt_schema import DOCUMENT_FIELDS, SPLITS, document_split, parquet_schema


DEFAULT_REPO_NAME = "moviegpt"
PLACEHOLDER_REPO_ID = "YOUR_USERNAME/moviegpt"
LOADER_GROUP_FUNCTION = """def group_texts(batch):
    concatenated = {key: sum(batch[key], []) for key in batch}
    usable = len(concatenated["input_ids"]) // block_size * block_size
    result = {
        key: [
            values[index:index + block_size]
            for index in range(0, usable, block_size)
        ]
        for key, values in concatenated.items()
    }
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result
"""


def collect_friends_documents(friends_dir: Path) -> list[dict[str, object]]:
    validation = validate_friends(friends_dir)
    documents: list[dict[str, object]] = []

    for _, manifest_record in iter_jsonl(friends_dir / "manifest.jsonl"):
        episode_id = manifest_record.get("episode_id")
        relative_file = manifest_record.get("file")
        season = manifest_record.get("season")
        episode = manifest_record.get("episode")
        if not isinstance(episode_id, str) or not isinstance(relative_file, str):
            raise ValueError("Invalid Friends manifest record")
        if not isinstance(season, int) or not isinstance(episode, int):
            raise ValueError(f"Invalid season or episode for {episode_id}")

        episode_path = friends_dir / relative_file
        text = episode_path.read_text(encoding="utf-8")
        if not text.endswith(f"{END_OF_TEXT}\n"):
            raise ValueError(f"Missing GPT-2 end marker in {episode_id}")

        documents.append(
            {
                "document_id": f"friends:{episode_id}",
                "source": "friends",
                "source_id": episode_id,
                "title": f"Friends S{season:02d}E{episode:02d}",
                "language": "en",
                "license": "Apache-2.0",
                "text": text,
                "text_sha256": sha256_file(episode_path),
                "character_count": len(text),
                "byte_count": episode_path.stat().st_size,
                "line_count": manifest_record.get("training_lines"),
            }
        )

    if len(documents) != validation["episodes"]:
        raise ValueError("Friends document count does not match validation")
    return sorted(documents, key=lambda record: str(record["document_id"]))


def collect_movie_documents(movie_dir: Path) -> list[dict[str, object]]:
    validation = validate_movies(movie_dir)
    documents: list[dict[str, object]] = []
    for _, manifest_record in iter_jsonl(movie_dir / "manifest.jsonl"):
        movie_id = manifest_record.get("movie_id")
        movie_name = manifest_record.get("movie_name")
        relative_file = manifest_record.get("file")
        if not all(
            isinstance(value, str)
            for value in (movie_id, movie_name, relative_file)
        ):
            raise ValueError("Invalid Cornell Movie-Dialogs manifest record")
        movie_path = movie_dir / relative_file
        text = movie_path.read_text(encoding="utf-8")
        if not text.endswith(f"{END_OF_TEXT}\n"):
            raise ValueError(f"Missing GPT-2 end marker in {movie_id}")
        documents.append(
            {
                "document_id": f"cornell_movie_dialogs:{movie_id}",
                "source": "cornell_movie_dialogs",
                "source_id": movie_id,
                "title": movie_name,
                "language": "en",
                "license": "other",
                "text": text,
                "text_sha256": sha256_file(movie_path),
                "character_count": len(text),
                "byte_count": movie_path.stat().st_size,
                "line_count": manifest_record.get("training_lines"),
            }
        )
    if len(documents) != validation["movies"]:
        raise ValueError("Cornell movie document count does not match validation")
    return sorted(documents, key=lambda record: str(record["document_id"]))


def write_parquet(path: Path, records: list[dict[str, object]]) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(records, schema=parquet_schema())
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )
    return {
        "bytes": path.stat().st_size,
        "documents": len(records),
        "file": f"data/{path.name}",
        "sha256": sha256_file(path),
    }


def dataset_card(repo_id: str, summary: dict[str, object]) -> str:
    split_counts = summary["splits"]
    source_counts = summary["sources"]
    opensubtitles_count = source_counts.get("opensubtitles_en", {}).get("documents", 0)
    return f"""---
license: other
language:
- en
pretty_name: MovieGPT
task_categories:
- text-generation
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
---

# MovieGPT

An English-language, document-oriented dialogue corpus prepared for causal
language-model training. All sources use one shared schema and one set of
train/validation/test splits. The `source` field records provenance without
separating the corpora into different dataset configurations.

## Current data

| Source | Documents | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Friends | {source_counts['friends']['documents']} | — | — | — |
| Cornell Movie-Dialogs | {source_counts['cornell_movie_dialogs']['documents']} | — | — | — |
| OpenSubtitles v2024 English | {opensubtitles_count} | — | — | — |
| **Unified total** | **{summary['documents']}** | **{split_counts['train']}** | **{split_counts['validation']}** | **{split_counts['test']}** |

Every row is one complete source document with a stable `document_id` and a
`text` field. Friends documents correspond to episodes, Cornell documents to
movies, and OpenSubtitles documents to artificial 512-line chunks because the
OPUS plain-text archive does not preserve movie boundaries. Utterances are
separated by newlines and every document ends with GPT-2's `<|endoftext|>`
token. Train, validation, and test assignment is a stable SHA-256-based 90/5/5
split, so adding sources later will not reshuffle existing documents.

## Load for GPT-2

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
block_size = min(tokenizer.model_max_length, 1024)

tokenized = dataset.map(
    lambda batch: tokenizer(batch["text"], add_special_tokens=False),
    batched=True,
    remove_columns=dataset["train"].column_names,
)

{LOADER_GROUP_FUNCTION}

language_model_data = tokenized.map(group_texts, batched=True)
```

Use `dataset.filter(lambda row: row["source"] == "friends")` when a single
source is needed. The Hub repository intentionally exposes one unified
configuration.

## Fields

- `document_id`: globally unique, stable ID prefixed by source.
- `source` and `source_id`: corpus provenance.
- `title`, `language`, and `license`: document metadata.
- `text`: GPT-2-ready document text ending in `<|endoftext|>`.
- `text_sha256`, `character_count`, `byte_count`, and `line_count`: integrity
  and auditing fields.

## Cleaning and provenance

Friends excludes transcript notes and nonverbal-only rows, removes
stage directions and transcript-parser artifacts, preserves scene boundaries,
and does not serialize speaker or annotation metadata into the training text.
Cornell Movie-Dialogs reconstructs each reply chain, removes empty utterances
and known screenplay-formatting tags, and preserves conversation boundaries.
OpenSubtitles removes subtitle markup, nonverbal cues, music-only lines,
caption credits, and immediate duplicates before fixed-size chunking.

The Friends source is ConvoKit's
[Friends Corpus](https://convokit.cornell.edu/documentation/friends.html),
whose documentation identifies the original Emory NLP data as Apache-2.0.
Cornell data comes from ConvoKit's
[Cornell Movie-Dialogs Corpus](https://convokit.cornell.edu/documentation/movie.html),
and subtitle text comes from
[OPUS OpenSubtitles v2024](https://opus.nlpl.eu/datasets/OpenSubtitles).
Licensing is recorded per document because the sources use different terms.
Users remain responsible for reviewing source-specific rights before
redistribution or model release.

Build metadata and counts are recorded in `dataset_summary.json`.
"""


def build_moviegpt_package(
    friends_dir: Path,
    output_dir: Path,
    repo_id: str = PLACEHOLDER_REPO_ID,
    movie_dir: Path = Path("data/processed/movie_corpus"),
    opensubtitles_dir: Path | None = None,
) -> dict[str, object]:
    friends_documents = collect_friends_documents(friends_dir)
    movie_documents = collect_movie_documents(movie_dir)
    core_documents = friends_documents + movie_documents
    records_by_split = {split: [] for split in SPLITS}
    for document in core_documents:
        records_by_split[document_split(str(document["document_id"]))].append(document)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for legacy_directory in (data_dir / "friends", data_dir / "movie_corpus"):
        if legacy_directory.exists():
            shutil.rmtree(legacy_directory)
    for stale_path in data_dir.glob("*-core.parquet"):
        stale_path.unlink()

    file_records: list[dict[str, object]] = []
    core_split_counts: dict[str, int] = {}
    for split in SPLITS:
        split_path = data_dir / f"{split}-core.parquet"
        records = sorted(records_by_split[split], key=lambda row: str(row["document_id"]))
        file_record = write_parquet(split_path, records)
        file_record["split"] = split
        file_record["kind"] = "core"
        file_records.append(file_record)
        core_split_counts[split] = len(records)

    opensubtitles_summary: dict[str, object] | None = None
    resolved_opensubtitles_dir = opensubtitles_dir or data_dir
    opensubtitles_summary_path = resolved_opensubtitles_dir / "opensubtitles_summary.json"
    if opensubtitles_summary_path.is_file():
        if resolved_opensubtitles_dir.resolve() != data_dir.resolve():
            raise ValueError(
                "OpenSubtitles shards must be generated directly in the package data directory"
            )
        validate_opensubtitles(resolved_opensubtitles_dir, opensubtitles_summary_path)
        with opensubtitles_summary_path.open(encoding="utf-8") as handle:
            opensubtitles_summary = json.load(handle)
        for source_file in opensubtitles_summary["files"]:
            file_records.append(
                {
                    **source_file,
                    "file": f"data/{source_file['file']}",
                    "kind": "opensubtitles",
                }
            )

    opensubtitles_splits = (
        opensubtitles_summary["splits"]
        if opensubtitles_summary is not None
        else {split: 0 for split in SPLITS}
    )
    split_counts = {
        split: core_split_counts[split] + int(opensubtitles_splits[split])
        for split in SPLITS
    }
    source_counts = {
        "friends": {"documents": len(friends_documents)},
        "cornell_movie_dialogs": {"documents": len(movie_documents)},
        "opensubtitles_en": {
            "documents": (
                int(opensubtitles_summary["cleaning"]["documents"])
                if opensubtitles_summary is not None
                else 0
            )
        },
    }

    summary: dict[str, object] = {
        "format_version": 2,
        "documents": sum(split_counts.values()),
        "splits": split_counts,
        "sources": source_counts,
        "files": file_records,
        "fields": list(DOCUMENT_FIELDS),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        dataset_card(repo_id, summary), encoding="utf-8"
    )
    return summary


def resolve_repo_id(repo_id: str | None, api: Any) -> str:
    if repo_id and "/" in repo_id:
        return repo_id
    namespace = api.whoami().get("name")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Hugging Face account has no usable namespace")
    return f"{namespace}/{repo_id or DEFAULT_REPO_NAME}"


def upload_moviegpt_package(
    package_dir: Path,
    repo_id: str | None = None,
    private: bool = True,
    api: Any | None = None,
) -> dict[str, str]:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    resolved_repo_id = resolve_repo_id(repo_id, api)
    repo_url = api.create_repo(
        resolved_repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=resolved_repo_id,
        repo_type="dataset",
        folder_path=package_dir,
        commit_message="Upload GPT-2-ready MovieGPT dataset",
        delete_patterns=["data/friends/**"],
    )
    return {"repo_id": resolved_repo_id, "url": str(repo_url)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--friends-dir",
        type=Path,
        default=Path("data/processed/friends"),
        help="Validated Friends corpus directory",
    )
    parser.add_argument(
        "--movie-dir",
        type=Path,
        default=Path("data/processed/movie_corpus"),
        help="Validated Cornell Movie-Dialogs corpus directory",
    )
    parser.add_argument(
        "--opensubtitles-dir",
        type=Path,
        help="OpenSubtitles Parquet directory; must equal OUTPUT_DIR/data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/moviegpt"),
        help="Generated Hugging Face dataset folder",
    )
    parser.add_argument(
        "--repo-id",
        help="Hub destination as namespace/moviegpt; inferred when authenticated",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Create the dataset repository and upload the generated folder",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository instead of the safer private default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = None
    resolved_repo_id = args.repo_id or PLACEHOLDER_REPO_ID
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        resolved_repo_id = resolve_repo_id(args.repo_id, api)

    summary = build_moviegpt_package(
        args.friends_dir,
        args.output_dir,
        repo_id=resolved_repo_id,
        movie_dir=args.movie_dir,
        opensubtitles_dir=args.opensubtitles_dir,
    )
    result: dict[str, object] = {"package": summary}
    if args.upload:
        result["hub"] = upload_moviegpt_package(
            args.output_dir,
            repo_id=resolved_repo_id,
            private=not args.public,
            api=api,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
