import importlib.util
import tempfile
import unittest
from pathlib import Path

from scripts.publish_moviegpt_dataset import (
    LOADER_GROUP_FUNCTION,
    build_moviegpt_package,
    document_split,
    upload_moviegpt_package,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


class FakeHfApi:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict[str, object]]] = []
        self.uploaded: list[dict[str, object]] = []

    def whoami(self) -> dict[str, str]:
        return {"name": "moviegpt-test"}

    def create_repo(self, repo_id: str, **kwargs: object) -> str:
        self.created.append((repo_id, kwargs))
        return f"https://huggingface.co/datasets/{repo_id}"

    def upload_folder(self, **kwargs: object) -> None:
        self.uploaded.append(kwargs)


class PublishMoviegptDatasetTest(unittest.TestCase):
    @unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow is required for package tests")
    def test_builds_loader_ready_hub_package(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "moviegpt"
            summary = build_moviegpt_package(
                PROJECT_ROOT / "data" / "processed" / "friends",
                output_dir,
                repo_id="example/moviegpt",
                movie_dir=PROJECT_ROOT / "data" / "processed" / "movie_corpus",
                allow_partial=True,
            )

            self.assertEqual(summary["documents"], 853)
            self.assertEqual(
                summary["splits"],
                {"train": 761, "validation": 49, "test": 43},
            )
            self.assertEqual(summary["sources"]["friends"]["documents"], 236)
            self.assertEqual(
                summary["sources"]["cornell_movie_dialogs"]["documents"], 617
            )

            records = []
            for split, expected_count in summary["splits"].items():
                split_path = output_dir / "data" / f"{split}-core.parquet"
                split_records = pq.read_table(split_path).to_pylist()
                self.assertEqual(len(split_records), expected_count)
                self.assertTrue(
                    all(document_split(record["document_id"]) == split for record in split_records)
                )
                records.extend(split_records)

            self.assertEqual(len({record["document_id"] for record in records}), 853)
            self.assertTrue(all(record["text"].endswith("<|endoftext|>\n") for record in records))
            self.assertEqual(
                {record["source"] for record in records},
                {"friends", "cornell_movie_dialogs"},
            )
            self.assertEqual(
                set(records[0]),
                {
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
                },
            )

            readme = (output_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn('load_dataset("example/moviegpt")', readme)
            self.assertIn("config_name: default", readme)
            self.assertNotIn("config_name: friends", readme)
            self.assertFalse((output_dir / "data" / "friends").exists())
            self.assertTrue((output_dir / "dataset_summary.json").is_file())

    @unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow is required for package tests")
    def test_requires_complete_opensubtitles_for_release_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "moviegpt"

            with self.assertRaisesRegex(
                FileNotFoundError, "Missing required OpenSubtitles summary"
            ):
                build_moviegpt_package(
                    PROJECT_ROOT / "data" / "processed" / "friends",
                    output_dir,
                    repo_id="example/moviegpt",
                    movie_dir=PROJECT_ROOT / "data" / "processed" / "movie_corpus",
                )

    def test_upload_defaults_to_private_user_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_dir = Path(temporary_directory)
            (package_dir / "README.md").write_text("dataset", encoding="utf-8")
            api = FakeHfApi()

            result = upload_moviegpt_package(package_dir, api=api)

            self.assertEqual(result["repo_id"], "moviegpt-test/moviegpt")
            self.assertEqual(
                api.created,
                [
                    (
                        "moviegpt-test/moviegpt",
                        {
                            "repo_type": "dataset",
                            "private": True,
                            "exist_ok": True,
                        },
                    )
                ],
            )
            self.assertEqual(api.uploaded[0]["repo_type"], "dataset")
            self.assertEqual(api.uploaded[0]["repo_id"], "moviegpt-test/moviegpt")
            self.assertEqual(api.uploaded[0]["delete_patterns"], ["data/friends/**"])

    def test_dataset_card_loader_chunks_every_tokenizer_column(self) -> None:
        namespace = {"block_size": 4}
        exec(LOADER_GROUP_FUNCTION, namespace)

        grouped = namespace["group_texts"](
            {
                "input_ids": [[10, 11, 12], [13, 14, 15]],
                "attention_mask": [[1, 1, 1], [1, 1, 1]],
            }
        )

        self.assertEqual(grouped["input_ids"], [[10, 11, 12, 13]])
        self.assertEqual(grouped["attention_mask"], [[1, 1, 1, 1]])
        self.assertEqual(grouped["labels"], grouped["input_ids"])


if __name__ == "__main__":
    unittest.main()
