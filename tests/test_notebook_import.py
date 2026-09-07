"""A Colab kernel may already have imported the unrelated PyPI model_hub package."""
import subprocess
import sys
import unittest
from pathlib import Path


class NotebookImportTests(unittest.TestCase):
    def test_training_import_with_unrelated_model_hub_already_loaded(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys
import types
unrelated = types.ModuleType("model_hub")
unrelated.__file__ = "/usr/local/lib/python3.13/dist-packages/model_hub/__init__.py"
sys.modules["model_hub"] = unrelated
import train_gpt2
assert callable(train_gpt2.export_model)
assert callable(train_gpt2.upload_export)
'''], cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
