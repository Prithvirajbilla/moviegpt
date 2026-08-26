#!/usr/bin/env python3
"""Stream-clean the OPUS OpenSubtitles English text archive into Parquet.

The OPUS plain-text archive does not retain subtitle-file or movie boundaries.
This cleaner therefore creates explicitly labeled artificial documents of a
fixed number of retained subtitle lines. Output shards are compressed Parquet
files suitable for Hugging Face streaming and are written without duplicating
the uncompressed source archive.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import shutil
import sys
import time
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .clean_friends_corpus import END_OF_TEXT, sha256_file
    from .moviegpt_schema import DOCUMENT_FIELDS, SPLITS, document_split, parquet_schema
else:
    from clean_friends_corpus import END_OF_TEXT, sha256_file
    from moviegpt_schema import DOCUMENT_FIELDS, SPLITS, document_split, parquet_schema


DEFAULT_LINES_PER_DOCUMENT = 512
DEFAULT_SHARD_TARGET_BYTES = 512 * 1024 * 1024
DEFAULT_ROW_GROUP_DOCUMENTS = 4096
DEFAULT_PROGRESS_EVERY_LINES = 10_000_000
DEFAULT_MIN_FREE_BYTES = 1024 * 1024 * 1024
DEFAULT_WORKERS = 4
DEFAULT_BATCH_LINES = 50_000
VALIDATION_READ_ATTEMPTS = 5
FORMATTING_TAG_RE = re.compile(r"<[^>\n]{0,200}>")
ASS_TAG_RE = re.compile(r"\{\\[^{}]{1,200}\}")
BRACKETED_CUE_RE = re.compile(r"\[[^\[\]\n]{0,200}\]|\([^()\n]{0,200}\)")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LEADING_SPEAKER_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 .&'’_-]{1,30}|[A-Z][a-z]{1,20}):\s*"
)
LEADING_DIALOGUE_MARK_RE = re.compile(r"^(?:-|–|—)\s+")
TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?\s*"
    r"(?:-->|-)?\s*\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?\s*$"
)
ARTIFACT_RE = re.compile(
    r"(?:https?://|www\.|opensubtitles|subscene|addic7ed|yify)"
    r"|\b(?:captions?|subtitles?|sync(?:ed|hronized)?)\s+(?:by|from)\b",
    re.IGNORECASE,
)
MOJIBAKE_REPLACEMENTS = {
    "âª": "♪",
    "â™ª": "♪",
    "â«": "♫",
    "â™«": "♫",
    "â€”": "—",
    "â€“": "–",
    "â€¦": "…",
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€": "”",
    "Â ": " ",
}


@dataclass
class LineCleaningResult:
    text: str | None
    reason: str | None = None
    formatting_tags_removed: int = 0
    bracketed_cues_removed: int = 0
    speaker_labels_removed: int = 0
    mojibake_repairs: int = 0
    replacement_characters: int = 0
    reserved_markers_removed: int = 0


@dataclass
class CleaningStats:
    source_bytes: int = 0
    source_lines: int = 0
    kept_lines: int = 0
    documents: int = 0
    excluded_blank: int = 0
    excluded_artifact: int = 0
    excluded_cue_only: int = 0
    excluded_music: int = 0
    excluded_timestamp_or_index: int = 0
    excluded_consecutive_duplicate: int = 0
    formatting_tags_removed: int = 0
    bracketed_cues_removed: int = 0
    speaker_labels_removed: int = 0
    mojibake_repairs: int = 0
    replacement_characters: int = 0
    reserved_markers_removed: int = 0

    def absorb(self, other: "CleaningStats") -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def record(self, result: LineCleaningResult) -> None:
        for name in (
            "formatting_tags_removed",
            "bracketed_cues_removed",
            "speaker_labels_removed",
            "mojibake_repairs",
            "replacement_characters",
            "reserved_markers_removed",
        ):
            setattr(self, name, getattr(self, name) + getattr(result, name))
        if result.text is not None:
            self.kept_lines += 1
        elif result.reason is not None:
            counter_name = f"excluded_{result.reason}"
            setattr(self, counter_name, getattr(self, counter_name) + 1)


def clean_line_batch(raw_lines: list[bytes]) -> tuple[list[str], CleaningStats]:
    """Clean an ordered batch in a worker process without changing line order."""

    cleaned_lines: list[str] = []
    stats = CleaningStats()
    for raw_bytes in raw_lines:
        stats.source_bytes += len(raw_bytes)
        stats.source_lines += 1
        result = clean_subtitle_line(raw_bytes.decode("utf-8", errors="replace"))
        stats.record(result)
        if result.text is not None:
            cleaned_lines.append(result.text)
    return cleaned_lines, stats


def sha256_file_resilient(path: Path) -> str:
    """Hash a shard while retrying transient macOS/cloud-backed read timeouts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        consecutive_timeouts = 0
        while True:
            try:
                chunk = handle.read(1024 * 1024)
            except TimeoutError:
                consecutive_timeouts += 1
                if consecutive_timeouts >= VALIDATION_READ_ATTEMPTS:
                    raise
                time.sleep(consecutive_timeouts)
                continue
            consecutive_timeouts = 0
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def bounded_ordered_map(
    executor: concurrent.futures.Executor,
    batches: Any,
    max_pending: int,
) -> Any:
    """Yield ordered worker results without eagerly consuming the input file."""

    batch_iterator = iter(batches)
    pending: deque[concurrent.futures.Future[Any]] = deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(clean_line_batch, next(batch_iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(clean_line_batch, next(batch_iterator)))
        except StopIteration:
            pass


def clean_subtitle_line(raw_line: str) -> LineCleaningResult:
    replacement_characters = raw_line.count("\ufffd")
    line = raw_line.strip().lstrip("\ufeff")
    if not line:
        return LineCleaningResult(
            text=None,
            reason="blank",
            replacement_characters=replacement_characters,
        )
    if TIMESTAMP_RE.fullmatch(line) or line.isdecimal():
        return LineCleaningResult(
            text=None,
            reason="timestamp_or_index",
            replacement_characters=replacement_characters,
        )

    if "&" in line:
        line = html.unescape(line)
    mojibake_repairs = 0
    if "â" in line or "Â" in line:
        for broken, repaired in MOJIBAKE_REPLACEMENTS.items():
            occurrences = line.count(broken)
            if occurrences:
                line = line.replace(broken, repaired)
                mojibake_repairs += occurrences

    if "♪" in line or "♫" in line:
        return LineCleaningResult(
            text=None,
            reason="music",
            mojibake_repairs=mojibake_repairs,
            replacement_characters=replacement_characters,
        )

    line, ass_tags_removed = ASS_TAG_RE.subn(" ", line)
    line, html_tags_removed = FORMATTING_TAG_RE.subn(" ", line)
    line, bracketed_cues_removed = BRACKETED_CUE_RE.subn(" ", line)
    speaker_labels_removed = 0
    if LEADING_SPEAKER_RE.match(line.lstrip()):
        line, speaker_labels_removed = LEADING_SPEAKER_RE.subn("", line.lstrip(), count=1)
    line = LEADING_DIALOGUE_MARK_RE.sub("", line.lstrip(), count=1)
    reserved_markers_removed = line.count(END_OF_TEXT)
    if reserved_markers_removed:
        line = line.replace(END_OF_TEXT, " ")
    line = CONTROL_CHARACTER_RE.sub(" ", line)
    line = " ".join(line.split())
    if line and not line.isascii():
        line = unicodedata.normalize("NFC", line)

    common = {
        "formatting_tags_removed": ass_tags_removed + html_tags_removed,
        "bracketed_cues_removed": bracketed_cues_removed,
        "speaker_labels_removed": speaker_labels_removed,
        "mojibake_repairs": mojibake_repairs,
        "replacement_characters": replacement_characters,
        "reserved_markers_removed": reserved_markers_removed,
    }
    if not line:
        return LineCleaningResult(text=None, reason="cue_only", **common)
    if ARTIFACT_RE.search(line):
        return LineCleaningResult(text=None, reason="artifact", **common)
    return LineCleaningResult(text=line, **common)


class ParquetSplitWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        shard_target_bytes: int,
        row_group_documents: int,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.pq = pq
        self.output_dir = output_dir
        self.split = split
        self.shard_target_bytes = shard_target_bytes
        self.row_group_documents = row_group_documents
        self.schema = parquet_schema()
        self.buffer: list[dict[str, object]] = []
        self.buffer_text_bytes = 0
        self.writer: Any | None = None
        self.partial_path: Path | None = None
        self.final_path: Path | None = None
        self.shard_index = 0
        self.shard_documents = 0
        self.shard_text_bytes = 0
        self.files: list[dict[str, object]] = []

    def add(self, record: dict[str, object]) -> None:
        self.buffer.append(record)
        self.buffer_text_bytes += int(record["byte_count"])
        if (
            len(self.buffer) >= self.row_group_documents
            or self.buffer_text_bytes >= 64 * 1024 * 1024
        ):
            self.flush()

    def _open(self) -> None:
        if self.writer is not None:
            return
        basename = f"{self.split}-opensubtitles-{self.shard_index:05d}.parquet"
        self.final_path = self.output_dir / basename
        self.partial_path = self.output_dir / f"{basename}.partial"
        self.writer = self.pq.ParquetWriter(
            self.partial_path,
            self.schema,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
        )

    def flush(self) -> None:
        if not self.buffer:
            return
        self._open()
        table = self.pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table, row_group_size=len(self.buffer))
        self.shard_documents += len(self.buffer)
        self.shard_text_bytes += self.buffer_text_bytes
        self.buffer.clear()
        self.buffer_text_bytes = 0
        if self.shard_text_bytes >= self.shard_target_bytes:
            self.close_shard()

    def close_shard(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        assert self.partial_path is not None and self.final_path is not None
        self.partial_path.replace(self.final_path)
        self.files.append(
            {
                "bytes": self.final_path.stat().st_size,
                "documents": self.shard_documents,
                "file": self.final_path.name,
                "sha256": sha256_file(self.final_path),
                "split": self.split,
                "text_bytes": self.shard_text_bytes,
            }
        )
        self.shard_index += 1
        self.shard_documents = 0
        self.shard_text_bytes = 0
        self.writer = None
        self.partial_path = None
        self.final_path = None

    def finish(self) -> list[dict[str, object]]:
        self.flush()
        self.close_shard()
        return self.files


def subtitle_document(chunk_index: int, lines: list[str]) -> dict[str, object]:
    source_id = f"chunk-{chunk_index:09d}"
    document_id = f"opensubtitles_en:{source_id}"
    text = "\n".join(lines) + f"\n{END_OF_TEXT}\n"
    encoded = text.encode("utf-8")
    return {
        "document_id": document_id,
        "source": "opensubtitles_en",
        "source_id": source_id,
        "title": f"OpenSubtitles v2024 English chunk {chunk_index:09d}",
        "language": "en",
        "license": "other",
        "text": text,
        "text_sha256": hashlib.sha256(encoded).hexdigest(),
        "character_count": len(text),
        "byte_count": len(encoded),
        "line_count": len(lines),
    }


def clean_opensubtitles(
    input_path: Path,
    output_dir: Path,
    lines_per_document: int = DEFAULT_LINES_PER_DOCUMENT,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    row_group_documents: int = DEFAULT_ROW_GROUP_DOCUMENTS,
    max_source_lines: int | None = None,
    start_byte: int = 0,
    progress_every_lines: int = DEFAULT_PROGRESS_EVERY_LINES,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    workers: int = 1,
    batch_lines: int = DEFAULT_BATCH_LINES,
) -> dict[str, object]:
    if lines_per_document <= 0:
        raise ValueError("lines_per_document must be positive")
    if shard_target_bytes <= 0 or row_group_documents <= 0:
        raise ValueError("shard settings must be positive")
    if start_byte < 0 or start_byte >= input_path.stat().st_size:
        raise ValueError("start_byte must point inside the source file")
    if progress_every_lines <= 0 or min_free_bytes < 0:
        raise ValueError("progress and free-space settings must be nonnegative")
    if workers <= 0 or batch_lines <= 0:
        raise ValueError("workers and batch_lines must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*-opensubtitles-*.parquet*"):
        stale_path.unlink()

    writers = {
        split: ParquetSplitWriter(
            output_dir,
            split,
            shard_target_bytes=shard_target_bytes,
            row_group_documents=row_group_documents,
        )
        for split in SPLITS
    }
    stats = CleaningStats()
    split_documents = {split: 0 for split in SPLITS}
    current_lines: list[str] = []
    previous_line: str | None = None
    chunk_index = 0
    input_digest = hashlib.sha256()
    input_complete = start_byte == 0
    started_at = time.monotonic()
    next_progress_line = progress_every_lines

    def emit_document() -> None:
        nonlocal chunk_index, current_lines
        if not current_lines:
            return
        record = subtitle_document(chunk_index, current_lines)
        split = document_split(str(record["document_id"]))
        writers[split].add(record)
        split_documents[split] += 1
        stats.documents += 1
        chunk_index += 1
        current_lines = []

    def source_batches() -> Any:
        nonlocal input_complete
        source_lines_read = 0
        with input_path.open("rb") as handle:
            if start_byte:
                handle.seek(start_byte)
                handle.readline()
            while True:
                batch: list[bytes] = []
                while len(batch) < batch_lines:
                    if (
                        max_source_lines is not None
                        and source_lines_read >= max_source_lines
                    ):
                        input_complete = False
                        break
                    raw_bytes = handle.readline()
                    if not raw_bytes:
                        break
                    input_digest.update(raw_bytes)
                    source_lines_read += 1
                    batch.append(raw_bytes)
                if batch:
                    yield batch
                if len(batch) < batch_lines:
                    break

    def consume_batch(batch_result: tuple[list[str], CleaningStats]) -> None:
        nonlocal previous_line, next_progress_line
        cleaned_lines, batch_stats = batch_result
        stats.absorb(batch_stats)
        for cleaned_line in cleaned_lines:
            if cleaned_line == previous_line:
                stats.kept_lines -= 1
                stats.excluded_consecutive_duplicate += 1
                continue
            previous_line = cleaned_line
            current_lines.append(cleaned_line)
            if len(current_lines) == lines_per_document:
                emit_document()

        if stats.source_lines >= next_progress_line:
            free_bytes = shutil.disk_usage(output_dir).free
            elapsed = time.monotonic() - started_at
            print(
                json.dumps(
                    {
                        "documents": stats.documents,
                        "elapsed_seconds": round(elapsed, 1),
                        "free_gib": round(free_bytes / (1024**3), 2),
                        "kept_lines": stats.kept_lines,
                        "lines_per_second": round(stats.source_lines / elapsed),
                        "source_lines": stats.source_lines,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            next_progress_line += progress_every_lines
            if free_bytes < min_free_bytes:
                raise OSError(
                    f"Stopping with only {free_bytes} free bytes in {output_dir}"
                )

    try:
        batches = source_batches()
        if workers == 1:
            for batch_result in map(clean_line_batch, batches):
                consume_batch(batch_result)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers
            ) as executor:
                for batch_result in bounded_ordered_map(
                    executor, batches, max_pending=workers * 2
                ):
                    consume_batch(batch_result)
        emit_document()
    except BaseException:
        for writer in writers.values():
            if writer.writer is not None:
                writer.writer.close()
        raise

    files = [file for split in SPLITS for file in writers[split].finish()]
    summary: dict[str, object] = {
        "cleaning": asdict(stats),
        "document_boundary": {
            "kind": "artificial_fixed_retained_lines",
            "lines_per_document": lines_per_document,
            "source_file_boundaries_available": False,
        },
        "fields": list(DOCUMENT_FIELDS),
        "files": files,
        "format_version": 1,
        "input": {
            "complete": input_complete,
            "file": input_path.name,
            "processed_bytes": stats.source_bytes,
            "processed_sha256": input_digest.hexdigest(),
            "start_byte": start_byte,
            "source_file_bytes": input_path.stat().st_size,
        },
        "splits": split_documents,
    }
    summary_path = output_dir / "opensubtitles_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_opensubtitles(output_dir, summary_path)
    return summary


def validate_opensubtitles(
    output_dir: Path, summary_path: Path | None = None
) -> dict[str, int]:
    import pyarrow.parquet as pq

    summary_path = summary_path or output_dir / "opensubtitles_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    files = summary.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Summary has no Parquet files")
    expected_schema = parquet_schema()
    total_documents = 0
    split_documents = {split: 0 for split in SPLITS}
    expected_names: set[str] = set()
    for file_record in files:
        if not isinstance(file_record, dict) or not isinstance(file_record.get("file"), str):
            raise ValueError("Invalid file record")
        filename = file_record["file"]
        relative_path = Path(filename)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe shard path {filename!r}")
        expected_names.add(filename)
        shard_path = output_dir / relative_path
        if shard_path.stat().st_size != file_record.get("bytes"):
            raise ValueError(f"Byte-count mismatch for {filename}")
        if sha256_file_resilient(shard_path) != file_record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {filename}")
        parquet_file = pq.ParquetFile(shard_path)
        if not parquet_file.schema_arrow.equals(expected_schema):
            raise ValueError(f"Schema mismatch for {filename}")
        documents = parquet_file.metadata.num_rows
        if documents != file_record.get("documents"):
            raise ValueError(f"Document-count mismatch for {filename}")
        split = file_record.get("split")
        if split not in split_documents:
            raise ValueError(f"Invalid split for {filename}")
        split_documents[split] += documents
        total_documents += documents
        sample = parquet_file.read_row_group(0).slice(0, 1).to_pylist()[0]
        if not sample["text"].endswith(f"{END_OF_TEXT}\n"):
            raise ValueError(f"Missing end marker in {filename}")
        if hashlib.sha256(sample["text"].encode("utf-8")).hexdigest() != sample["text_sha256"]:
            raise ValueError(f"Text SHA-256 mismatch in {filename}")
    actual_names = {path.name for path in output_dir.glob("*-opensubtitles-*.parquet")}
    if actual_names != expected_names:
        raise ValueError("Parquet shards do not match the summary")
    if split_documents != summary.get("splits"):
        raise ValueError("Split counts do not match the summary")
    cleaning = summary.get("cleaning")
    if not isinstance(cleaning, dict) or cleaning.get("documents") != total_documents:
        raise ValueError("Cleaning document count does not match the shards")
    return {"documents": total_documents, **split_documents}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/raw/opensubtitles/OpenSubtitles-v2024-en.txt"),
        help="English OPUS OpenSubtitles plain-text archive",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/opensubtitles"),
        help="Directory for compressed Parquet shards",
    )
    parser.add_argument("--lines-per-document", type=int, default=DEFAULT_LINES_PER_DOCUMENT)
    parser.add_argument("--shard-target-bytes", type=int, default=DEFAULT_SHARD_TARGET_BYTES)
    parser.add_argument("--row-group-documents", type=int, default=DEFAULT_ROW_GROUP_DOCUMENTS)
    parser.add_argument(
        "--max-source-lines",
        type=int,
        help="Process only the first N source lines for benchmarking",
    )
    parser.add_argument(
        "--start-byte",
        type=int,
        default=0,
        help="Seek to a source-byte offset and discard the first partial line",
    )
    parser.add_argument(
        "--progress-every-lines",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_LINES,
        help="Write a progress record after each N source lines",
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=DEFAULT_MIN_FREE_BYTES,
        help="Stop before output storage falls below this many free bytes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel worker processes for pure subtitle-line cleaning",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=DEFAULT_BATCH_LINES,
        help="Ordered source lines sent to each cleaning worker",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing generated shards instead of regenerating them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = (
        validate_opensubtitles(args.output_dir)
        if args.check
        else clean_opensubtitles(
            args.input_file,
            args.output_dir,
            lines_per_document=args.lines_per_document,
            shard_target_bytes=args.shard_target_bytes,
            row_group_documents=args.row_group_documents,
            max_source_lines=args.max_source_lines,
            start_byte=args.start_byte,
            progress_every_lines=args.progress_every_lines,
            min_free_bytes=args.min_free_bytes,
            workers=args.workers,
            batch_lines=args.batch_lines,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
