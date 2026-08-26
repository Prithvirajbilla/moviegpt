#!/usr/bin/env python3
"""Convert the ConvoKit Cornell Movie-Dialogs corpus into movie documents.

Each output file contains one normalized spoken utterance per line, a blank
line between source conversations, and an ``<|endoftext|>`` marker at the end
of the movie. Conversation turns are reconstructed from ``reply-to`` links;
the source JSONL does not store them in chronological order.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

if __package__:
    from .clean_friends_corpus import (
        END_OF_TEXT,
        artifact_metadata,
        iter_jsonl,
        normalize_text,
        sha256_file,
        write_jsonl,
    )
else:
    from clean_friends_corpus import (  # type: ignore[no-redef]
        END_OF_TEXT,
        artifact_metadata,
        iter_jsonl,
        normalize_text,
        sha256_file,
        write_jsonl,
    )


LINE_ID_RE = re.compile(r"^L(?P<number>\d+)$")
MOVIE_ID_RE = re.compile(r"^m(?P<number>\d+)$")
FORMATTING_TAG_RE = re.compile(r"</?(?:b|html|i|pre|u)\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class ConversationMetadata:
    conversation_id: str
    movie_id: str
    movie_name: str
    release_year: str
    rating: str
    votes: str
    genres: list[str]


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    reply_to: str | None
    text: str


@dataclass
class Movie:
    movie_id: str
    movie_name: str
    release_year: str
    rating: str
    votes: str
    genres: list[str]
    source_url: str | None
    conversations: dict[str, list[Utterance]] = field(default_factory=dict)
    source_conversations: int = 0
    source_utterances: int = 0
    excluded_empty: int = 0
    excluded_empty_conversations: int = 0
    formatting_tags_removed: int = 0

    @property
    def training_lines(self) -> int:
        return sum(len(conversation) for conversation in self.conversations.values())


def numeric_id(identifier: str, pattern: re.Pattern[str], label: str) -> int:
    match = pattern.fullmatch(identifier)
    if match is None:
        raise ValueError(f"Unexpected {label} id {identifier!r}")
    return int(match.group("number"))


def parse_genres(value: object, conversation_id: str) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"Invalid genres for conversation {conversation_id!r}")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"Invalid genres for conversation {conversation_id!r}"
        ) from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Invalid genres for conversation {conversation_id!r}")
    return parsed


def clean_dialogue_text(text: object) -> tuple[str, int]:
    if not isinstance(text, str):
        return "", 0
    without_tags, formatting_tags_removed = FORMATTING_TAG_RE.subn("", text)
    return normalize_text(without_tags), formatting_tags_removed


def load_conversation_metadata(path: Path) -> dict[str, ConversationMetadata]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Expected conversations.json to contain an object")

    conversations: dict[str, ConversationMetadata] = {}
    for conversation_id, record in raw.items():
        numeric_id(conversation_id, LINE_ID_RE, "conversation")
        meta = record.get("meta") if isinstance(record, dict) else None
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid metadata for conversation {conversation_id!r}")
        required = {
            key: meta.get(key)
            for key in (
                "movie_idx",
                "movie_name",
                "release_year",
                "rating",
                "votes",
            )
        }
        if not all(isinstance(value, str) for value in required.values()):
            raise ValueError(f"Invalid metadata for conversation {conversation_id!r}")
        movie_id = required["movie_idx"]
        numeric_id(movie_id, MOVIE_ID_RE, "movie")
        conversations[conversation_id] = ConversationMetadata(
            conversation_id=conversation_id,
            movie_id=movie_id,
            movie_name=required["movie_name"],
            release_year=required["release_year"],
            rating=required["rating"],
            votes=required["votes"],
            genres=parse_genres(meta.get("genre"), conversation_id),
        )
    return conversations


def load_movie_urls(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    urls = raw.get("url") if isinstance(raw, dict) else None
    if not isinstance(urls, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in urls.items()
    ):
        raise ValueError("Invalid movie URLs in corpus.json")
    return urls


def order_conversation(
    conversation_id: str, utterances: dict[str, Utterance]
) -> list[Utterance]:
    roots = [utterance for utterance in utterances.values() if utterance.reply_to is None]
    if len(roots) != 1:
        raise ValueError(
            f"Conversation {conversation_id!r} has {len(roots)} root utterances"
        )

    children: dict[str, Utterance] = {}
    for utterance in utterances.values():
        if utterance.reply_to is None:
            continue
        if utterance.reply_to not in utterances:
            raise ValueError(
                f"Conversation {conversation_id!r} has missing parent "
                f"{utterance.reply_to!r}"
            )
        if utterance.reply_to in children:
            raise ValueError(
                f"Conversation {conversation_id!r} branches at "
                f"{utterance.reply_to!r}"
            )
        children[utterance.reply_to] = utterance

    ordered: list[Utterance] = []
    current: Utterance | None = roots[0]
    while current is not None:
        ordered.append(current)
        current = children.get(current.utterance_id)
        if current is not None and current in ordered:
            raise ValueError(f"Conversation {conversation_id!r} contains a cycle")
    if len(ordered) != len(utterances):
        raise ValueError(f"Conversation {conversation_id!r} is disconnected")
    return ordered


def collect_movies(
    input_dir: Path,
) -> tuple[
    dict[str, Movie],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    metadata = load_conversation_metadata(input_dir / "conversations.json")
    movie_urls = load_movie_urls(input_dir / "corpus.json")
    raw_conversations: dict[str, dict[str, Utterance]] = defaultdict(dict)
    formatting_tags_by_conversation: dict[str, int] = defaultdict(int)
    excluded_records: list[dict[str, object]] = []
    text_repair_records: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for line_number, record in iter_jsonl(input_dir / "utterances.jsonl"):
        utterance_id = record.get("id")
        conversation_id = record.get("conversation_id")
        raw_text = record.get("text")
        reply_to = record.get("reply-to")
        utterance_meta = record.get("meta")

        if not isinstance(utterance_id, str):
            raise ValueError(f"Missing utterance id on line {line_number}")
        numeric_id(utterance_id, LINE_ID_RE, "utterance")
        if utterance_id in seen_ids:
            raise ValueError(f"Duplicate utterance id {utterance_id!r}")
        seen_ids.add(utterance_id)
        if not isinstance(conversation_id, str) or conversation_id not in metadata:
            raise ValueError(
                f"Unknown conversation {conversation_id!r} for utterance {utterance_id!r}"
            )
        if reply_to is not None and not isinstance(reply_to, str):
            raise ValueError(f"Invalid reply-to value for utterance {utterance_id!r}")
        if isinstance(reply_to, str):
            numeric_id(reply_to, LINE_ID_RE, "reply-to")
        movie_id = (
            utterance_meta.get("movie_id")
            if isinstance(utterance_meta, dict)
            else None
        )
        if movie_id != metadata[conversation_id].movie_id:
            raise ValueError(f"Movie mismatch for utterance {utterance_id!r}")
        if END_OF_TEXT in (raw_text if isinstance(raw_text, str) else ""):
            raise ValueError(f"Reserved end marker found in utterance {utterance_id!r}")

        cleaned_text, formatting_tags_removed = clean_dialogue_text(raw_text)
        raw_conversations[conversation_id][utterance_id] = Utterance(
            utterance_id=utterance_id,
            reply_to=reply_to,
            text=cleaned_text,
        )
        if formatting_tags_removed:
            formatting_tags_by_conversation[conversation_id] += formatting_tags_removed
            text_repair_records.append(
                {
                    "cleaned_text": cleaned_text,
                    "conversation_id": conversation_id,
                    "formatting_tags_removed": formatting_tags_removed,
                    "movie_id": movie_id,
                    "source_text": raw_text,
                    "utterance_id": utterance_id,
                }
            )

    movies: dict[str, Movie] = {}
    for conversation_id in sorted(
        metadata, key=lambda value: numeric_id(value, LINE_ID_RE, "conversation")
    ):
        if conversation_id not in raw_conversations:
            raise ValueError(f"Conversation {conversation_id!r} has no utterances")
        conversation_meta = metadata[conversation_id]
        movie = movies.setdefault(
            conversation_meta.movie_id,
            Movie(
                movie_id=conversation_meta.movie_id,
                movie_name=conversation_meta.movie_name,
                release_year=conversation_meta.release_year,
                rating=conversation_meta.rating,
                votes=conversation_meta.votes,
                genres=conversation_meta.genres,
                source_url=movie_urls.get(conversation_meta.movie_id),
            ),
        )
        if (
            movie.movie_name != conversation_meta.movie_name
            or movie.release_year != conversation_meta.release_year
            or movie.rating != conversation_meta.rating
            or movie.votes != conversation_meta.votes
            or movie.genres != conversation_meta.genres
        ):
            raise ValueError(
                f"Inconsistent metadata for movie {conversation_meta.movie_id!r}"
            )

        movie.source_conversations += 1
        ordered = order_conversation(
            conversation_id, raw_conversations[conversation_id]
        )
        movie.source_utterances += len(ordered)
        kept: list[Utterance] = []
        for utterance in ordered:
            if utterance.text:
                kept.append(utterance)
            else:
                movie.excluded_empty += 1
                excluded_records.append(
                    {
                        "conversation_id": conversation_id,
                        "movie_id": movie.movie_id,
                        "reason": "empty",
                        "utterance_id": utterance.utterance_id,
                    }
                )
        if kept:
            movie.conversations[conversation_id] = kept
        else:
            movie.excluded_empty_conversations += 1

        movie.formatting_tags_removed += formatting_tags_by_conversation[conversation_id]

    if set(movie_urls) != set(movies):
        missing_urls = sorted(set(movies) - set(movie_urls))
        unknown_urls = sorted(set(movie_urls) - set(movies))
        raise ValueError(
            f"Movie URL mismatch: missing={missing_urls[:3]!r}, "
            f"unknown={unknown_urls[:3]!r}"
        )
    return movies, text_repair_records, excluded_records


def render_movie(movie: Movie) -> str:
    conversation_blocks = [
        "\n".join(utterance.text for utterance in movie.conversations[conversation_id])
        for conversation_id in sorted(
            movie.conversations,
            key=lambda value: numeric_id(value, LINE_ID_RE, "conversation"),
        )
    ]
    return "\n\n".join(conversation_blocks) + f"\n{END_OF_TEXT}\n"


def clean_movie_corpus(input_dir: Path, output_dir: Path) -> dict[str, object]:
    movies, text_repair_records, excluded_records = collect_movies(input_dir)
    movies_dir = output_dir / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, object]] = []
    expected_movie_files: set[Path] = set()

    for movie_id in sorted(
        movies, key=lambda value: numeric_id(value, MOVIE_ID_RE, "movie")
    ):
        movie = movies[movie_id]
        movie_path = movies_dir / f"{movie_id}.txt"
        movie_path.write_text(render_movie(movie), encoding="utf-8")
        expected_movie_files.add(movie_path)
        manifest_records.append(
            {
                "bytes": movie_path.stat().st_size,
                "conversations": len(movie.conversations),
                "excluded_empty": movie.excluded_empty,
                "excluded_empty_conversations": movie.excluded_empty_conversations,
                "file": f"movies/{movie_id}.txt",
                "genres": movie.genres,
                "movie_id": movie_id,
                "movie_name": movie.movie_name,
                "rating": movie.rating,
                "release_year": movie.release_year,
                "sha256": sha256_file(movie_path),
                "source_url": movie.source_url,
                "source_conversations": movie.source_conversations,
                "source_utterances": movie.source_utterances,
                "formatting_tags_removed": movie.formatting_tags_removed,
                "training_lines": movie.training_lines,
                "votes": movie.votes,
            }
        )

    for stale_path in movies_dir.glob("m*.txt"):
        if stale_path not in expected_movie_files:
            stale_path.unlink()

    write_jsonl(output_dir / "manifest.jsonl", manifest_records)
    write_jsonl(output_dir / "text_repairs.jsonl", text_repair_records)
    write_jsonl(output_dir / "excluded_utterances.jsonl", excluded_records)
    source_files: dict[str, dict[str, object]] = {}
    for source_name in ("utterances.jsonl", "conversations.json", "corpus.json"):
        source_path = input_dir / source_name
        source_files[source_name] = {
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }
    summary: dict[str, object] = {
        "conversations": sum(
            int(record["conversations"]) for record in manifest_records
        ),
        "excluded_empty": len(excluded_records),
        "excluded_empty_conversations": sum(
            int(record["excluded_empty_conversations"])
            for record in manifest_records
        ),
        "formatting_tags_removed": sum(
            int(record["formatting_tags_removed"]) for record in manifest_records
        ),
        "generated_artifacts": {
            "excluded_utterances.jsonl": artifact_metadata(
                output_dir / "excluded_utterances.jsonl", len(excluded_records)
            ),
            "manifest.jsonl": artifact_metadata(
                output_dir / "manifest.jsonl", len(manifest_records)
            ),
            "text_repairs.jsonl": artifact_metadata(
                output_dir / "text_repairs.jsonl", len(text_repair_records)
            ),
        },
        "movies": len(manifest_records),
        "source_files": source_files,
        "source_conversations": sum(
            int(record["source_conversations"]) for record in manifest_records
        ),
        "source_utterances": sum(
            int(record["source_utterances"]) for record in manifest_records
        ),
        "training_lines": sum(
            int(record["training_lines"]) for record in manifest_records
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_cleaned_corpus(output_dir)
    return summary


def validate_cleaned_corpus(output_dir: Path) -> dict[str, int]:
    manifest_records = [
        record for _, record in iter_jsonl(output_dir / "manifest.jsonl")
    ]
    seen_files: set[str] = set()
    totals = {
        "conversations": 0,
        "excluded_empty": 0,
        "excluded_empty_conversations": 0,
        "formatting_tags_removed": 0,
        "source_conversations": 0,
        "source_utterances": 0,
        "training_lines": 0,
    }
    for record in manifest_records:
        relative_name = record.get("file")
        if not isinstance(relative_name, str):
            raise ValueError("Manifest record has no file name")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe manifest path {relative_name!r}")
        if relative_name in seen_files:
            raise ValueError(f"Duplicate manifest path {relative_name!r}")
        seen_files.add(relative_name)

        movie_path = output_dir / relative_path
        if movie_path.stat().st_size != record.get("bytes"):
            raise ValueError(f"Byte-count mismatch for {relative_name}")
        if sha256_file(movie_path) != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {relative_name}")
        movie_text = movie_path.read_text(encoding="utf-8")
        end_marker = f"{END_OF_TEXT}\n"
        if not movie_text.endswith(end_marker):
            raise ValueError(f"Missing end marker in {relative_name}")
        movie_body = movie_text[: -len(end_marker)]
        if END_OF_TEXT in movie_body or not movie_body.endswith("\n"):
            raise ValueError(f"Invalid end-marker layout in {relative_name}")
        conversation_blocks = movie_body[:-1].split("\n\n")
        if any(
            not block or any(not line for line in block.splitlines())
            for block in conversation_blocks
        ):
            raise ValueError(f"Invalid conversation layout in {relative_name}")
        training_lines = sum(
            len(block.splitlines()) for block in conversation_blocks
        )
        if training_lines != record.get("training_lines"):
            raise ValueError(f"Training-line mismatch for {relative_name}")
        if len(conversation_blocks) != record.get("conversations"):
            raise ValueError(f"Conversation-count mismatch for {relative_name}")
        for key in totals:
            totals[key] += int(record[key])

    actual_files = {
        str(path.relative_to(output_dir))
        for path in (output_dir / "movies").glob("m*.txt")
    }
    if seen_files != actual_files:
        raise ValueError("Movie files do not match the manifest")

    with (output_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    expected_counts = {"movies": len(manifest_records), **totals}
    for name, expected in expected_counts.items():
        if summary.get(name) != expected:
            raise ValueError(f"Summary mismatch for {name}")

    generated_artifacts = summary.get("generated_artifacts")
    expected_artifacts = {
        "excluded_utterances.jsonl",
        "manifest.jsonl",
        "text_repairs.jsonl",
    }
    if not isinstance(generated_artifacts, dict) or set(generated_artifacts) != expected_artifacts:
        raise ValueError("Generated artifacts do not match the summary")
    for artifact_name in sorted(expected_artifacts):
        metadata = generated_artifacts[artifact_name]
        artifact_path = output_dir / artifact_name
        if artifact_path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"Byte-count mismatch for {artifact_name}")
        if sha256_file(artifact_path) != metadata.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {artifact_name}")
        records = sum(1 for _ in iter_jsonl(artifact_path))
        if records != metadata.get("records"):
            raise ValueError(f"Record-count mismatch for {artifact_name}")

    return {"movies": len(manifest_records), "training_lines": totals["training_lines"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/convokit/movie-corpus"),
        help="Directory containing the ConvoKit Cornell Movie-Dialogs files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/movie_corpus"),
        help="Directory for movie documents and manifests",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing generated files instead of regenerating them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = (
        validate_cleaned_corpus(args.output_dir)
        if args.check
        else clean_movie_corpus(args.input_dir, args.output_dir)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
