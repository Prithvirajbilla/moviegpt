#!/usr/bin/env python3
"""Convert the ConvoKit Friends corpus into GPT-2 episode documents.

The ConvoKit release calls its source file ``utterances.jsonl``. Each output
file contains one spoken utterance per line, a blank line between scenes, and
an ``<|endoftext|>`` marker at the end of the episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable


END_OF_TEXT = "<|endoftext|>"
TRANSCRIPT_NOTE_SPEAKER = "TRANSCRIPT_NOTE"
EXTRA_SPEAKER_ALIASES = {
    "Chandler",
    "Charlie",
    "Frank Jr.",
    "Joey",
    "Monica",
    "Phoebe",
    "Rachel",
    "Ross",
    "Susan",
    "TV",
}
EPISODE_ID_RE = re.compile(
    r"^s(?P<season>\d{2})_e(?P<episode>\d{2})_c(?P<scene>\d{2})_u(?P<turn>\d+)$"
)
STAGE_DIRECTION_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
ARTIFACT_MARKER_RE = re.compile(
    r"\*?\b(?:opening|closing|end)\s+credits\b\*?\.?(?:\s*)"
    r"|\bopening\s+sequence\b\.?(?:\s*)"
    r"|\bcommercial\s+break\b\.?(?:\s*)"
    r"|(?:^|(?<=[.!?]))\s*(?:gap\s+)?commercial(?:[.!?]|$)\s*"
    r"|(?:^|(?<=[.!?]))\s*-?cut\s+to\s+[^.!?]*[.!?]?\s*"
    r"|\bfade\s+(?:in|out)\b\.?(?:\s*)",
    re.IGNORECASE,
)
ORPHAN_COLON_RE = re.compile(r"(?:^:\s*|\s+:\s*)")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;!?])")


@dataclass
class CleaningStats:
    source_utterances: int = 0
    kept_source_utterances: int = 0
    recovered_from_transcript_note: int = 0
    stage_directions_removed: int = 0
    artifact_markers_removed: int = 0
    embedded_speaker_labels_removed: int = 0
    orphan_colons_removed: int = 0
    excluded_transcript_notes: int = 0
    excluded_empty: int = 0

    def absorb(self, other: "CleaningStats") -> None:
        for stats_field in fields(self):
            setattr(
                self,
                stats_field.name,
                getattr(self, stats_field.name) + getattr(other, stats_field.name),
            )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class TextCleaningResult:
    lines: list[str]
    stage_directions_removed: int
    artifact_markers_removed: int
    embedded_speaker_labels: list[str]
    orphan_colons_removed: int

    @property
    def changed(self) -> bool:
        return any(
            (
                self.stage_directions_removed,
                self.artifact_markers_removed,
                self.embedded_speaker_labels,
                self.orphan_colons_removed,
            )
        )


@dataclass
class Episode:
    season: str
    episode: str
    scenes: dict[str, list[tuple[int, int, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    stats: CleaningStats = field(default_factory=CleaningStats)

    @property
    def training_lines(self) -> int:
        return sum(len(turns) for turns in self.scenes.values())


def normalize_text(text: str) -> str:
    """Apply conservative normalization while preserving punctuation and case."""

    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_stage_directions(text: str) -> tuple[str, int]:
    removed = 0
    while True:
        text, substitutions = STAGE_DIRECTION_RE.subn(" ", text)
        removed += substitutions
        if substitutions == 0:
            break

    unmatched_open = re.search(r"[\[(]", text)
    if unmatched_open is not None:
        text = text[: unmatched_open.start()]
        removed += 1

    for closing_character in (")", "]"):
        while closing_character in text:
            closing_index = text.index(closing_character)
            if closing_index > 0 and text[closing_index - 1].isdigit():
                text = (
                    text[:closing_index]
                    + "__NUMBERED_CLOSE__"
                    + text[closing_index + 1 :]
                )
                continue
            prefix = text[:closing_index]
            sentence_boundaries = [
                prefix.rfind(punctuation) for punctuation in (".", "!", "?")
            ]
            stage_start = max(sentence_boundaries)
            retained_prefix = prefix[: stage_start + 1] if stage_start >= 0 else ""
            text = retained_prefix + " " + text[closing_index + 1 :]
            removed += 1

    return text.replace("__NUMBERED_CLOSE__", ")"), removed


def load_conversation_metadata(path: Path) -> dict[str, tuple[str, str, str]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    conversations: dict[str, tuple[str, str, str]] = {}
    for conversation_id, metadata in raw.items():
        season = metadata.get("season")
        episode = metadata.get("episode")
        scene = metadata.get("scene")
        if not all(isinstance(value, str) for value in (season, episode, scene)):
            raise ValueError(f"Invalid metadata for conversation {conversation_id!r}")
        conversations[conversation_id] = (season, episode, scene)
    return conversations


def load_speaker_aliases(path: Path) -> set[str]:
    """Build conservative aliases used to remove leaked transcript labels."""

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    aliases = set(EXTRA_SPEAKER_ALIASES)
    for speaker_name in raw:
        if speaker_name == TRANSCRIPT_NOTE_SPEAKER:
            continue
        aliases.add(speaker_name)

    return aliases


def compile_speaker_label_pattern(aliases: set[str]) -> re.Pattern[str]:
    escaped_aliases = sorted((re.escape(alias) for alias in aliases), key=len, reverse=True)
    return re.compile(
        rf"(?:^|(?<=[.!?]))[ \t]*({'|'.join(escaped_aliases)})\s*:\s*",
        re.IGNORECASE,
    )


def split_embedded_speaker_labels(
    text: str, pattern: re.Pattern[str]
) -> tuple[list[str], list[str]]:
    """Split transcript-parser leaks such as ``Maybe. Joey: Wait.``"""

    lines: list[str] = []
    labels: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        before_label = text[cursor : match.start()].strip()
        if before_label:
            lines.append(before_label)
        labels.append(match.group(1))
        cursor = match.end()

    if not labels:
        return [text], []

    after_last_label = text[cursor:].strip()
    if after_last_label:
        lines.append(after_last_label)
    return lines, labels


def split_orphan_colons(text: str) -> tuple[list[str], int]:
    matches = list(ORPHAN_COLON_RE.finditer(text))
    if not matches:
        return [text], 0
    return [part for part in ORPHAN_COLON_RE.split(text) if part.strip()], len(matches)


def clean_training_text(
    text: str, speaker_label_pattern: re.Pattern[str]
) -> TextCleaningResult:
    text, stage_directions_removed = strip_stage_directions(text)
    text, artifact_markers_removed = ARTIFACT_MARKER_RE.subn(" ", text)
    text = normalize_text(text)

    speaker_segments, embedded_speaker_labels = split_embedded_speaker_labels(
        text, speaker_label_pattern
    )
    lines: list[str] = []
    orphan_colons_removed = 0
    for segment in speaker_segments:
        colon_segments, removed = split_orphan_colons(segment)
        orphan_colons_removed += removed
        for colon_segment in colon_segments:
            cleaned_segment = normalize_text(colon_segment)
            cleaned_segment = SPACE_BEFORE_PUNCTUATION_RE.sub(
                r"\1", cleaned_segment
            )
            if cleaned_segment:
                lines.append(cleaned_segment)

    return TextCleaningResult(
        lines=lines,
        stage_directions_removed=stage_directions_removed,
        artifact_markers_removed=artifact_markers_removed,
        embedded_speaker_labels=embedded_speaker_labels,
        orphan_colons_removed=orphan_colons_removed,
    )


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            yield line_number, record


def collect_episodes(
    input_dir: Path,
) -> tuple[
    dict[tuple[str, str], Episode],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    conversations = load_conversation_metadata(input_dir / "conversations.json")
    speaker_aliases = load_speaker_aliases(input_dir / "speakers.json")
    speaker_label_pattern = compile_speaker_label_pattern(speaker_aliases)
    episodes: dict[tuple[str, str], Episode] = {}
    text_repair_records: list[dict[str, object]] = []
    excluded_records: list[dict[str, object]] = []
    seen_utterance_ids: set[str] = set()

    for line_number, utterance in iter_jsonl(input_dir / "utterances.jsonl"):
        utterance_id = utterance.get("id")
        conversation_id = utterance.get("conversation_id")
        speaker = utterance.get("speaker")
        raw_text = utterance.get("text")
        meta = utterance.get("meta")
        transcript_with_note = (
            meta.get("transcript_with_note") if isinstance(meta, dict) else None
        )

        if not isinstance(utterance_id, str):
            raise ValueError(f"Missing utterance id on line {line_number}")
        if utterance_id in seen_utterance_ids:
            raise ValueError(f"Duplicate utterance id {utterance_id!r}")
        seen_utterance_ids.add(utterance_id)
        id_match = EPISODE_ID_RE.fullmatch(utterance_id)
        if id_match is None:
            raise ValueError(f"Unexpected utterance id {utterance_id!r}")
        if not isinstance(conversation_id, str) or conversation_id not in conversations:
            raise ValueError(
                f"Unknown conversation {conversation_id!r} for utterance {utterance_id!r}"
            )

        season, episode_number, scene = conversations[conversation_id]
        if (
            season != f"s{id_match.group('season')}"
            or episode_number != f"e{id_match.group('episode')}"
            or scene != f"c{id_match.group('scene')}"
        ):
            raise ValueError(f"ID/metadata mismatch for utterance {utterance_id!r}")

        episode_key = (season, episode_number)
        episode = episodes.setdefault(
            episode_key, Episode(season=season, episode=episode_number)
        )
        episode.stats.source_utterances += 1

        if speaker == TRANSCRIPT_NOTE_SPEAKER:
            episode.stats.excluded_transcript_notes += 1
            excluded_records.append(
                {
                    "utterance_id": utterance_id,
                    "speaker": speaker,
                    "reason": "transcript_note",
                    "source_text": raw_text,
                    "transcript_with_note": transcript_with_note,
                }
            )
            continue

        primary_text = normalize_text(raw_text) if isinstance(raw_text, str) else ""
        annotated_text = (
            normalize_text(transcript_with_note)
            if isinstance(transcript_with_note, str)
            else ""
        )
        text = annotated_text or primary_text
        used_transcript_with_note = bool(annotated_text)
        recovered_from_transcript_note = bool(annotated_text and not primary_text)

        if END_OF_TEXT in text:
            raise ValueError(f"Reserved end marker found in utterance {utterance_id!r}")

        cleaning_result = clean_training_text(
            text, speaker_label_pattern
        )
        if not cleaning_result.lines:
            episode.stats.excluded_empty += 1
            excluded_records.append(
                {
                    "utterance_id": utterance_id,
                    "speaker": speaker,
                    "reason": "empty_or_nonverbal",
                    "source_text": raw_text,
                    "transcript_with_note": transcript_with_note,
                }
            )
            continue

        episode.stats.kept_source_utterances += 1
        episode.stats.recovered_from_transcript_note += int(
            recovered_from_transcript_note
        )
        episode.stats.stage_directions_removed += (
            cleaning_result.stage_directions_removed
        )
        episode.stats.artifact_markers_removed += (
            cleaning_result.artifact_markers_removed
        )
        episode.stats.embedded_speaker_labels_removed += len(
            cleaning_result.embedded_speaker_labels
        )
        episode.stats.orphan_colons_removed += cleaning_result.orphan_colons_removed
        turn = int(id_match.group("turn"))
        for segment, training_line in enumerate(cleaning_result.lines):
            episode.scenes[scene].append((turn, segment, training_line))

        if recovered_from_transcript_note or cleaning_result.changed:
            text_repair_records.append(
                {
                    "utterance_id": utterance_id,
                    "speaker": speaker,
                    "source_text": raw_text,
                    "transcript_with_note": transcript_with_note,
                    "used_transcript_with_note": used_transcript_with_note,
                    "recovered_from_transcript_note": (
                        recovered_from_transcript_note
                    ),
                    "stage_directions_removed": (
                        cleaning_result.stage_directions_removed
                    ),
                    "artifact_markers_removed": (
                        cleaning_result.artifact_markers_removed
                    ),
                    "embedded_speaker_labels_removed": (
                        cleaning_result.embedded_speaker_labels
                    ),
                    "orphan_colons_removed": cleaning_result.orphan_colons_removed,
                    "cleaned_lines": cleaning_result.lines,
                }
            )

    return episodes, text_repair_records, excluded_records


def render_episode(episode: Episode) -> str:
    scene_blocks: list[str] = []
    for scene in sorted(episode.scenes):
        turns = sorted(episode.scenes[scene], key=lambda item: (item[0], item[1]))
        scene_blocks.append("\n".join(text for _, _, text in turns))
    return "\n\n".join(scene_blocks) + f"\n{END_OF_TEXT}\n"


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def artifact_metadata(path: Path, records: int) -> dict[str, object]:
    return {
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def clean_friends_corpus(input_dir: Path, output_dir: Path) -> dict[str, object]:
    episodes, text_repair_records, excluded_records = collect_episodes(input_dir)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    expected_episode_files: set[Path] = set()

    for episode_key in sorted(episodes):
        episode = episodes[episode_key]
        episode_name = f"{episode.season}{episode.episode}.txt"
        episode_path = episodes_dir / episode_name
        episode_path.write_text(render_episode(episode), encoding="utf-8")
        expected_episode_files.add(episode_path)

        manifest_records.append(
            {
                "episode_id": f"{episode.season}{episode.episode}",
                "season": int(episode.season[1:]),
                "episode": int(episode.episode[1:]),
                "file": f"episodes/{episode_name}",
                "bytes": episode_path.stat().st_size,
                "sha256": sha256_file(episode_path),
                "scenes": len(episode.scenes),
                "training_lines": episode.training_lines,
                **episode.stats.as_dict(),
            }
        )

    for stale_path in episodes_dir.glob("s??e??.txt"):
        if stale_path not in expected_episode_files:
            stale_path.unlink()

    generated_record_counts = {
        "manifest.jsonl": len(manifest_records),
        "text_repairs.jsonl": len(text_repair_records),
        "excluded_utterances.jsonl": len(excluded_records),
    }
    for artifact_name, records in (
        ("manifest.jsonl", manifest_records),
        ("text_repairs.jsonl", text_repair_records),
        ("excluded_utterances.jsonl", excluded_records),
    ):
        write_jsonl(output_dir / artifact_name, records)
    (output_dir / "embedded_speaker_labels.jsonl").unlink(missing_ok=True)

    corpus_stats = CleaningStats()
    for episode in episodes.values():
        corpus_stats.absorb(episode.stats)

    source_files = {}
    for source_name in ("utterances.jsonl", "conversations.json", "speakers.json"):
        source_path = input_dir / source_name
        source_files[source_name] = {
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }

    summary: dict[str, object] = {
        "episodes": len(manifest_records),
        "scenes": sum(int(record["scenes"]) for record in manifest_records),
        "training_lines": sum(
            int(record["training_lines"]) for record in manifest_records
        ),
        **corpus_stats.as_dict(),
        "source_files": source_files,
        "generated_artifacts": {
            artifact_name: artifact_metadata(output_dir / artifact_name, record_count)
            for artifact_name, record_count in generated_record_counts.items()
        },
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
    training_lines = 0
    scenes = 0

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

        episode_path = output_dir / relative_path
        if episode_path.stat().st_size != record.get("bytes"):
            raise ValueError(f"Byte-count mismatch for {relative_name}")
        if sha256_file(episode_path) != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {relative_name}")
        episode_text = episode_path.read_text(encoding="utf-8")
        end_marker = f"{END_OF_TEXT}\n"
        if not episode_text.endswith(end_marker):
            raise ValueError(f"Missing end marker in {relative_name}")
        episode_body = episode_text[: -len(end_marker)]
        if END_OF_TEXT in episode_body:
            raise ValueError(f"Unexpected end marker in {relative_name}")
        if not episode_body.endswith("\n"):
            raise ValueError(f"Missing newline before end marker in {relative_name}")

        scene_blocks = episode_body[:-1].split("\n\n")
        if any(
            not block or any(not line for line in block.splitlines())
            for block in scene_blocks
        ):
            raise ValueError(f"Invalid scene layout in {relative_name}")
        actual_training_lines = sum(
            len(block.splitlines()) for block in scene_blocks
        )
        actual_scenes = len(scene_blocks)
        if actual_training_lines != record.get("training_lines"):
            raise ValueError(f"Training-line mismatch for {relative_name}")
        if actual_scenes != record.get("scenes"):
            raise ValueError(f"Scene-count mismatch for {relative_name}")
        training_lines += actual_training_lines
        scenes += actual_scenes

    actual_files = {
        str(path.relative_to(output_dir))
        for path in (output_dir / "episodes").glob("s??e??.txt")
    }
    if seen_files != actual_files:
        raise ValueError("Episode files do not match the manifest")

    with (output_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    expected_artifacts = summary.get("generated_artifacts")
    if not isinstance(expected_artifacts, dict):
        raise ValueError("Summary has no generated-artifact metadata")
    expected_artifact_names = {
        "manifest.jsonl",
        "text_repairs.jsonl",
        "excluded_utterances.jsonl",
    }
    if set(expected_artifacts) != expected_artifact_names:
        raise ValueError("Generated artifacts do not match the summary")
    for artifact_name in sorted(expected_artifact_names):
        metadata = expected_artifacts[artifact_name]
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata for {artifact_name}")
        artifact_path = output_dir / artifact_name
        if artifact_path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"Byte-count mismatch for {artifact_name}")
        if sha256_file(artifact_path) != metadata.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {artifact_name}")
        actual_records = sum(1 for _ in iter_jsonl(artifact_path))
        if actual_records != metadata.get("records"):
            raise ValueError(f"Record-count mismatch for {artifact_name}")

    expected_summary_counts = {
        "episodes": len(manifest_records),
        "scenes": scenes,
        "training_lines": training_lines,
    }
    for stats_field in fields(CleaningStats):
        expected_summary_counts[stats_field.name] = sum(
            int(record[stats_field.name]) for record in manifest_records
        )
    for name, expected_value in expected_summary_counts.items():
        if summary.get(name) != expected_value:
            raise ValueError(f"Summary mismatch for {name}")

    return {"episodes": len(manifest_records), "training_lines": training_lines}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/convokit/friends-corpus"),
        help="Directory containing ConvoKit utterances.jsonl and conversations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/friends"),
        help="Directory for episode text files and manifests",
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
        else clean_friends_corpus(args.input_dir, args.output_dir)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
