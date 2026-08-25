import json
import tempfile
import unittest
from pathlib import Path

from scripts.clean_friends_corpus import (
    clean_friends_corpus,
    clean_training_text,
    compile_speaker_label_pattern,
    normalize_text,
    split_embedded_speaker_labels,
    validate_cleaned_corpus,
)


class CleanFriendsCorpusTest(unittest.TestCase):
    def test_normalize_text_is_conservative(self) -> None:
        self.assertEqual(normalize_text("  Cafe\u0301\u00a0  Central  "), "Café Central")

    def test_only_splits_speaker_labels_at_dialogue_boundaries(self) -> None:
        pattern = compile_speaker_label_pattern({"Joey", "Ross"})
        self.assertEqual(
            split_embedded_speaker_labels("Maybe. Joey: Wait.", pattern),
            (["Maybe.", "Wait."], ["Joey"]),
        )
        self.assertEqual(
            split_embedded_speaker_labels(
                'First things first: touch my abs. "Ross: The Divorce-Force".',
                pattern,
            ),
            (
                ['First things first: touch my abs. "Ross: The Divorce-Force".'],
                [],
            ),
        )

    def test_cleans_stage_directions_artifacts_and_parser_colons(self) -> None:
        pattern = compile_speaker_label_pattern({"Joey"})
        result = clean_training_text(
            "Opening credits. : Hello (waves). Joey: Wait.", pattern
        )
        self.assertEqual(result.lines, ["Hello.", "Wait."])
        self.assertEqual(result.stage_directions_removed, 1)
        self.assertEqual(result.artifact_markers_removed, 1)
        self.assertEqual(result.embedded_speaker_labels, ["Joey"])
        self.assertEqual(result.orphan_colons_removed, 1)
        self.assertEqual(
            clean_training_text("Hello. [waves", pattern).lines,
            ["Hello."],
        )
        self.assertEqual(
            clean_training_text("Hello. She walks away)", pattern).lines,
            ["Hello."],
        )
        self.assertEqual(
            clean_training_text("Question 1) Why?", pattern).lines,
            ["Question 1) Why?"],
        )
        self.assertEqual(
            clean_training_text(
                "Mexico! End credits. Okay. -Cut to Rachel. Grandpa! Commercial",
                pattern,
            ).lines,
            ["Mexico! Okay. Grandpa!"],
        )

    def test_writes_one_clean_document_per_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            conversations = {
                "s01_e01_c01_u001": {
                    "season": "s01",
                    "episode": "e01",
                    "scene": "c01",
                },
                "s01_e01_c02_u001": {
                    "season": "s01",
                    "episode": "e01",
                    "scene": "c02",
                },
                "s01_e02_c01_u001": {
                    "season": "s01",
                    "episode": "e02",
                    "scene": "c01",
                },
            }
            (input_dir / "conversations.json").write_text(
                json.dumps(conversations), encoding="utf-8"
            )
            (input_dir / "speakers.json").write_text(
                json.dumps({"Joey Tribbiani": {}, "Rachel Green": {}}),
                encoding="utf-8",
            )

            utterances = [
                {
                    "id": "s01_e01_c01_u002",
                    "conversation_id": "s01_e01_c01_u001",
                    "speaker": "Joey Tribbiani",
                    "text": "  How you doin'? Joey: Wait. ",
                },
                {
                    "id": "s01_e01_c01_u001",
                    "conversation_id": "s01_e01_c01_u001",
                    "speaker": "Rachel Green",
                    "text": "Hi.",
                },
                {
                    "id": "s01_e01_c02_u001",
                    "conversation_id": "s01_e01_c02_u001",
                    "speaker": "TRANSCRIPT_NOTE",
                    "text": "[Scene changes]",
                },
                {
                    "id": "s01_e01_c02_u002",
                    "conversation_id": "s01_e01_c02_u001",
                    "speaker": "Monica Geller",
                    "text": "Welcome home.",
                },
                {
                    "id": "s01_e02_c01_u001",
                    "conversation_id": "s01_e02_c01_u001",
                    "speaker": "Ross Geller",
                    "text": "",
                    "meta": {"transcript_with_note": "(whispers) Phoebe?"},
                },
                {
                    "id": "s01_e02_c01_u002",
                    "conversation_id": "s01_e02_c01_u001",
                    "speaker": "Phoebe Buffay",
                    "text": "Smelly cat.",
                },
                {
                    "id": "s01_e02_c01_u003",
                    "conversation_id": "s01_e02_c01_u001",
                    "speaker": "Ross Geller",
                    "text": "",
                },
            ]
            with (input_dir / "utterances.jsonl").open("w", encoding="utf-8") as handle:
                for utterance in utterances:
                    handle.write(json.dumps(utterance) + "\n")

            summary = clean_friends_corpus(input_dir, output_dir)

            self.assertEqual(
                (output_dir / "episodes" / "s01e01.txt").read_text(
                    encoding="utf-8"
                ),
                "Hi.\nHow you doin'?\nWait.\n\nWelcome home.\n<|endoftext|>\n",
            )
            self.assertEqual(
                (output_dir / "episodes" / "s01e02.txt").read_text(
                    encoding="utf-8"
                ),
                "Phoebe?\nSmelly cat.\n<|endoftext|>\n",
            )
            expected_counts = {
                "episodes": 2,
                "scenes": 3,
                "source_utterances": 7,
                "kept_source_utterances": 5,
                "training_lines": 6,
                "recovered_from_transcript_note": 1,
                "stage_directions_removed": 1,
                "artifact_markers_removed": 0,
                "embedded_speaker_labels_removed": 1,
                "orphan_colons_removed": 0,
                "excluded_transcript_notes": 1,
                "excluded_empty": 1,
            }
            for key, value in expected_counts.items():
                self.assertEqual(summary[key], value)
            self.assertEqual(set(summary["source_files"]), {
                "utterances.jsonl",
                "conversations.json",
                "speakers.json",
            })
            self.assertEqual(
                set(summary["generated_artifacts"]),
                {
                    "manifest.jsonl",
                    "text_repairs.jsonl",
                    "excluded_utterances.jsonl",
                },
            )

            repair_records = [
                json.loads(line)
                for line in (output_dir / "text_repairs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(repair_records), 2)
            self.assertEqual(
                repair_records[0]["embedded_speaker_labels_removed"], ["Joey"]
            )
            self.assertEqual(
                repair_records[0]["cleaned_lines"], ["How you doin'?", "Wait."]
            )
            self.assertTrue(repair_records[1]["recovered_from_transcript_note"])
            self.assertEqual(repair_records[1]["cleaned_lines"], ["Phoebe?"])

            excluded_records = (output_dir / "excluded_utterances.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(excluded_records), 2)
            self.assertEqual(
                validate_cleaned_corpus(output_dir),
                {"episodes": 2, "training_lines": 6},
            )

            repairs_path = output_dir / "text_repairs.jsonl"
            original_repairs = repairs_path.read_text(encoding="utf-8")
            repairs_path.write_text(original_repairs + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Byte-count mismatch"):
                validate_cleaned_corpus(output_dir)
            repairs_path.write_text(original_repairs, encoding="utf-8")

            manifest_path = output_dir / "manifest.jsonl"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            manifest_records = [
                json.loads(line) for line in original_manifest.splitlines()
            ]
            manifest_records[0]["training_lines"] = 999
            manifest_path.write_text(
                "".join(json.dumps(record) + "\n" for record in manifest_records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Training-line mismatch"):
                validate_cleaned_corpus(output_dir)
            manifest_path.write_text(original_manifest, encoding="utf-8")

            episode_path = output_dir / "episodes" / "s01e01.txt"
            episode_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Byte-count mismatch"):
                validate_cleaned_corpus(output_dir)


if __name__ == "__main__":
    unittest.main()
