import json
import tempfile
import unittest
from pathlib import Path

from scripts.clean_movie_corpus import (
    clean_dialogue_text,
    clean_movie_corpus,
    order_conversation,
    Utterance,
    validate_cleaned_corpus,
)


class CleanMovieCorpusTest(unittest.TestCase):
    def test_strips_only_known_formatting_tags(self) -> None:
        self.assertEqual(
            clean_dialogue_text("That is <U>very</U> <i>good</i>. <breathing>"),
            ("That is very good. <breathing>", 4),
        )

    def test_rejects_branched_reply_chain(self) -> None:
        utterances = {
            "L1": Utterance("L1", None, "Root"),
            "L2": Utterance("L2", "L1", "First child"),
            "L3": Utterance("L3", "L1", "Second child"),
        }
        with self.assertRaisesRegex(ValueError, "branches"):
            order_conversation("L1", utterances)

    def test_writes_one_ordered_document_per_movie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            conversations = {
                "L20": {
                    "meta": {
                        "movie_idx": "m1",
                        "movie_name": "Second Film",
                        "release_year": "2002",
                        "rating": "7.1",
                        "votes": "11",
                        "genre": "['drama']",
                    }
                },
                "L10": {
                    "meta": {
                        "movie_idx": "m0",
                        "movie_name": "First Film",
                        "release_year": "2001",
                        "rating": "6.5",
                        "votes": "10",
                        "genre": "['comedy', 'romance']",
                    }
                },
                "L5": {
                    "meta": {
                        "movie_idx": "m0",
                        "movie_name": "First Film",
                        "release_year": "2001",
                        "rating": "6.5",
                        "votes": "10",
                        "genre": "['comedy', 'romance']",
                    }
                },
            }
            (input_dir / "conversations.json").write_text(
                json.dumps(conversations), encoding="utf-8"
            )
            (input_dir / "corpus.json").write_text(
                json.dumps(
                    {
                        "url": {
                            "m0": "https://example.test/first",
                            "m1": "https://example.test/second",
                        }
                    }
                ),
                encoding="utf-8",
            )
            utterances = [
                {
                    "id": "L11",
                    "conversation_id": "L10",
                    "text": "  <u>Cafe\u0301</u>   Central. ",
                    "reply-to": "L10",
                    "meta": {"movie_id": "m0"},
                },
                {
                    "id": "L10",
                    "conversation_id": "L10",
                    "text": "Later conversation.",
                    "reply-to": None,
                    "meta": {"movie_id": "m0"},
                },
                {
                    "id": "L6",
                    "conversation_id": "L5",
                    "text": "First reply.",
                    "reply-to": "L5",
                    "meta": {"movie_id": "m0"},
                },
                {
                    "id": "L5",
                    "conversation_id": "L5",
                    "text": "First root.",
                    "reply-to": None,
                    "meta": {"movie_id": "m0"},
                },
                {
                    "id": "L20",
                    "conversation_id": "L20",
                    "text": "Only line.",
                    "reply-to": None,
                    "meta": {"movie_id": "m1"},
                },
                {
                    "id": "L12",
                    "conversation_id": "L10",
                    "text": "   ",
                    "reply-to": "L11",
                    "meta": {"movie_id": "m0"},
                },
            ]
            with (input_dir / "utterances.jsonl").open("w", encoding="utf-8") as handle:
                for utterance in utterances:
                    handle.write(json.dumps(utterance) + "\n")

            summary = clean_movie_corpus(input_dir, output_dir)

            self.assertEqual(
                (output_dir / "movies" / "m0.txt").read_text(encoding="utf-8"),
                "First root.\nFirst reply.\n\nLater conversation.\nCafé Central.\n"
                "<|endoftext|>\n",
            )
            self.assertEqual(
                (output_dir / "movies" / "m1.txt").read_text(encoding="utf-8"),
                "Only line.\n<|endoftext|>\n",
            )
            self.assertEqual(summary["movies"], 2)
            self.assertEqual(summary["conversations"], 3)
            self.assertEqual(summary["source_conversations"], 3)
            self.assertEqual(summary["source_utterances"], 6)
            self.assertEqual(summary["training_lines"], 5)
            self.assertEqual(summary["excluded_empty"], 1)
            self.assertEqual(summary["excluded_empty_conversations"], 0)
            self.assertEqual(summary["formatting_tags_removed"], 2)
            self.assertEqual(
                validate_cleaned_corpus(output_dir),
                {"movies": 2, "training_lines": 5},
            )

            manifest_records = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest_records[0]["movie_id"], "m0")
            self.assertEqual(manifest_records[0]["genres"], ["comedy", "romance"])
            self.assertEqual(manifest_records[0]["conversations"], 2)
            repairs = [
                json.loads(line)
                for line in (output_dir / "text_repairs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(repairs[0]["utterance_id"], "L11")
            self.assertEqual(repairs[0]["formatting_tags_removed"], 2)
            excluded = json.loads(
                (output_dir / "excluded_utterances.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(excluded["utterance_id"], "L12")

            movie_path = output_dir / "movies" / "m0.txt"
            movie_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Byte-count mismatch"):
                validate_cleaned_corpus(output_dir)


if __name__ == "__main__":
    unittest.main()
