import importlib.util
import tempfile
import unittest
from pathlib import Path

from scripts.clean_opensubtitles import (
    clean_opensubtitles,
    clean_subtitle_line,
    validate_opensubtitles,
)


PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


class CleanOpenSubtitlesTest(unittest.TestCase):
    def test_cleans_subtitle_markup_and_cues(self) -> None:
        self.assertEqual(clean_subtitle_line("[Amy] <i>Hello&nbsp;there.</i>\n").text, "Hello there.")
        self.assertEqual(clean_subtitle_line("(laughing)\n").reason, "cue_only")
        self.assertEqual(clean_subtitle_line("âª Theme song âª\n").reason, "music")
        self.assertEqual(clean_subtitle_line("00:01:02,000 --> 00:01:03,500\n").reason, "timestamp_or_index")
        self.assertEqual(clean_subtitle_line("Subtitles by example.com\n").reason, "artifact")
        self.assertEqual(clean_subtitle_line("Question: why?\n").text, "why?")
        self.assertEqual(clean_subtitle_line("<breathing> Still here.\n").text, "Still here.")

    @unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow is required for Parquet tests")
    def test_streams_documents_to_flat_parquet_shards(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "subtitles.txt"
            output_dir = root / "output"
            input_path.write_text(
                "[Amy] Hello.\n"
                "Hello.\n"
                "<i>How are you?</i>\n"
                "(laughs)\n"
                "Fine.\n"
                "âª song âª\n"
                "Visit www.example.com\n"
                "- Goodbye.\n",
                encoding="utf-8",
            )

            summary = clean_opensubtitles(
                input_path,
                output_dir,
                lines_per_document=2,
                shard_target_bytes=1,
                row_group_documents=1,
                batch_lines=2,
            )
            cleaning = summary["cleaning"]
            self.assertEqual(cleaning["source_lines"], 8)
            self.assertEqual(cleaning["kept_lines"], 4)
            self.assertEqual(cleaning["excluded_consecutive_duplicate"], 1)
            self.assertEqual(cleaning["excluded_cue_only"], 1)
            self.assertEqual(cleaning["excluded_music"], 1)
            self.assertEqual(cleaning["excluded_artifact"], 1)
            self.assertEqual(cleaning["documents"], 2)

            shard_paths = sorted(output_dir.glob("*.parquet"))
            self.assertEqual(len(shard_paths), 2)
            rows = []
            for shard_path in shard_paths:
                rows.extend(pq.read_table(shard_path).to_pylist())
            rows.sort(key=lambda record: record["document_id"])
            self.assertEqual(
                [record["text"] for record in rows],
                [
                    "Hello.\nHow are you?\n<|endoftext|>\n",
                    "Fine.\nGoodbye.\n<|endoftext|>\n",
                ],
            )
            self.assertTrue(all(record["source"] == "opensubtitles_en" for record in rows))
            self.assertEqual(
                validate_opensubtitles(output_dir)["documents"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
