#!/usr/bin/env python3
"""Build and optionally upload the GPT-2-ready MovieGPT Hub dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .clean_friends_corpus import END_OF_TEXT, iter_jsonl, sha256_file, write_jsonl
    from .clean_friends_corpus import validate_cleaned_corpus
else:
    from clean_friends_corpus import END_OF_TEXT, iter_jsonl, sha256_file, write_jsonl
    from clean_friends_corpus import validate_cleaned_corpus


SPLITS = ("train", "validation", "test")
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


def document_split(document_id: str) -> str:
    """Assign a stable 90/5/5 split without depending on corpus order."""

    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16)
    percentile = bucket % 10_000
    if percentile < 9_000:
        return "train"
    if percentile < 9_500:
        return "validation"
    return "test"


def collect_friends_documents(friends_dir: Path) -> list[dict[str, object]]:
    validation = validate_cleaned_corpus(friends_dir)
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


def dataset_card(repo_id: str, summary: dict[str, object]) -> str:
    split_counts = summary["splits"]
    return f"""---
license: other
language:
- en
pretty_name: MovieGPT
task_categories:
- text-generation
configs:
- config_name: all
  default: true
  data_files:
  - split: train
    path: data/friends/train.jsonl
  - split: validation
    path: data/friends/validation.jsonl
  - split: test
    path: data/friends/test.jsonl
- config_name: friends
  data_files:
  - split: train
    path: data/friends/train.jsonl
  - split: validation
    path: data/friends/validation.jsonl
  - split: test
    path: data/friends/test.jsonl
---

# MovieGPT

An English-language, document-oriented dialogue corpus prepared for causal
language-model training. This repository will grow as additional movie and
subtitle sources are cleaned. The current release contains the cleaned
ConvoKit Friends corpus.

## Current data

| Source | Documents | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Friends | {summary['documents']} | {split_counts['train']} | {split_counts['validation']} | {split_counts['test']} |

Every row is one complete source document with a stable `document_id` and a
`text` field. Friends documents correspond to episodes. Utterances are
separated by newlines, scenes by blank lines, and every document ends with
GPT-2's `<|endoftext|>` token. Train, validation, and test assignment is a
stable SHA-256-based 90/5/5 split, so adding sources later will not reshuffle
existing documents.

## Load for GPT-2

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("{repo_id}", "all")
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

Use the `friends` configuration to load only that source. The default `all`
configuration currently contains the same records and will include additional
cleaned sources in future releases.

## Fields

- `document_id`: globally unique, stable ID prefixed by source.
- `source` and `source_id`: corpus provenance.
- `title`, `language`, and `license`: document metadata.
- `text`: GPT-2-ready document text ending in `<|endoftext|>`.
- `text_sha256`, `character_count`, `byte_count`, and `line_count`: integrity
  and auditing fields.

## Cleaning and provenance

The Friends data excludes transcript notes and nonverbal-only rows, removes
stage directions and transcript-parser artifacts, preserves scene boundaries,
and does not serialize speaker or annotation metadata into the training text.
The source is ConvoKit's
[Friends Corpus](https://convokit.cornell.edu/documentation/friends.html),
whose documentation identifies the original Emory NLP data as Apache-2.0.
Licensing is recorded per document because future MovieGPT sources may use
different licenses. Users remain responsible for reviewing source-specific
terms before redistribution or model release.

Build metadata and counts are recorded in `dataset_summary.json`.
"""


def build_moviegpt_package(
    friends_dir: Path,
    output_dir: Path,
    repo_id: str = PLACEHOLDER_REPO_ID,
) -> dict[str, object]:
    documents = collect_friends_documents(friends_dir)
    records_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for document in documents:
        records_by_split[document_split(str(document["document_id"]))].append(
            document
        )

    friends_output_dir = output_dir / "data" / "friends"
    friends_output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        split_path = friends_output_dir / f"{split}.jsonl"
        write_jsonl(split_path, records_by_split[split])
        split_files[split] = {
            "documents": len(records_by_split[split]),
            "bytes": split_path.stat().st_size,
            "sha256": sha256_file(split_path),
        }

    summary: dict[str, object] = {
        "format_version": 1,
        "documents": len(documents),
        "splits": {
            split: len(records_by_split[split]) for split in SPLITS
        },
        "sources": {"friends": {"documents": len(documents)}},
        "files": split_files,
        "fields": [
            "document_id",
            "source",
            "source_id",
            "title",
            "language",
            "license",
            "text",
            "text_sha256",
            "character_count",
            "byte_count",
            "line_count",
        ],
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
