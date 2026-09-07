import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from moviegpt_hub import export_model, upload_export
from train_gpt2 import GPT, GPTConfig


class ModelHubTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, 'hello': 2, 'world': 3},
                                                 unk_token='<unk>')),
            unk_token='<unk>', bos_token='<eos>', eos_token='<eos>',
        )
        self.model = GPT(GPTConfig(vocab_size=8, block_size=16, n_layer=1, n_head=2, n_embd=8))

    def export(self, path, log_file=None):
        return export_model(self.model, self.tokenizer, path, step=5000,
                            dataset_id='relentlessml/moviegpt', dataset_revision='fixed-sha',
                            val_loss=2.0, log_file=log_file)

    def test_transformers_round_trip_preserves_logits_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.export(directory)
            loaded = AutoModelForCausalLM.from_pretrained(folder).eval()
            tokenizer = AutoTokenizer.from_pretrained(folder)
            inputs = torch.tensor([[2, 3, 1, 2]])
            with torch.no_grad():
                original, _ = self.model(inputs)
                exported = loaded(inputs).logits
            torch.testing.assert_close(original, exported, atol=1e-5, rtol=1e-5)
            self.assertTrue(self.model.training)  # export must not change the training model
            self.assertEqual(tokenizer.get_vocab(), self.tokenizer.get_vocab())
            self.assertEqual(loaded.generation_config.suppress_tokens, [4, 5, 6, 7])
            self.assertTrue((folder / 'model.safetensors').is_file())
            metadata = json.loads((folder / 'training_metadata.json').read_text())
            self.assertEqual(metadata['optimizer_steps_completed'], 5000)
            self.assertEqual(metadata['dataset_revision'], 'fixed-sha')

    def test_upload_uses_model_repo_and_scoped_export(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.export(directory)
            with patch('moviegpt_hub.HfApi') as api:
                upload_export(folder, 'relentlessml/moviegpt', private=False)
                api.return_value.create_repo.assert_called_once_with(
                    repo_id='relentlessml/moviegpt', repo_type='model', private=False, exist_ok=True)
                kwargs = api.return_value.upload_folder.call_args.kwargs
                self.assertEqual(kwargs['repo_type'], 'model')
                self.assertEqual(kwargs['folder_path'], str(folder))
                self.assertNotIn('*.pt', kwargs['allow_patterns'])
                self.assertIn('5000', kwargs['commit_message'])

    def test_upload_failure_keeps_local_export(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.export(directory)
            with patch('moviegpt_hub.HfApi') as api:
                api.return_value.upload_folder.side_effect = RuntimeError('offline')
                with self.assertRaisesRegex(RuntimeError, 'offline'):
                    upload_export(folder, 'owner/repo')
            self.assertTrue((folder / 'model.safetensors').exists())

    def test_incomplete_export_never_contacts_hub(self):
        with tempfile.TemporaryDirectory() as directory, patch('moviegpt_hub.HfApi') as api:
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                upload_export(directory, 'owner/repo')
            api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
