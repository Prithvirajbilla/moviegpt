import unittest
from dataclasses import asdict

import torch

from train_gpt2 import (GPT, GPTConfig, MODEL_PRESETS, preset_config,
                        select_model_config, model_size_name, model_locations)


class ModelPresetTests(unittest.TestCase):
    def test_parameter_counts_and_attention_shapes(self):
        expected = {'small': 30357504, 'medium': 51500032, 'gpt2': 124475904}
        for name in MODEL_PRESETS:
            with self.subTest(preset=name), torch.device('meta'):
                config = preset_config(name)
                model = GPT(config)
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected[name])
                self.assertEqual(config.n_embd // config.n_head, 64)
                self.assertEqual(config.n_embd % config.n_head, 0)
                logits, _ = model(torch.ones((1, 4), dtype=torch.long))
                self.assertEqual(tuple(logits.shape), (1, 4, 50304))
                self.assertEqual(config.block_size, 1024)

    def test_default_is_current_gpt2_model(self):
        self.assertEqual(select_model_config(), GPTConfig(vocab_size=50304))
        self.assertEqual(model_locations('gpt2'), ('log', 'relentlessml/moviegpt'))

    def test_smaller_models_have_separate_locations(self):
        for name in ['small', 'medium']:
            self.assertEqual(model_locations(name), (f'log/{name}', f'relentlessml/moviegpt-{name}'))

    def test_resume_keeps_saved_architecture_and_rejects_conflict(self):
        for name in MODEL_PRESETS:
            checkpoint = {'config': asdict(preset_config(name))}
            self.assertEqual(select_model_config(checkpoint=checkpoint), preset_config(name))
            self.assertEqual(select_model_config(name, checkpoint), preset_config(name))
            other = 'small' if name != 'small' else 'gpt2'
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                select_model_config(other, checkpoint)
            self.assertEqual(model_size_name(select_model_config(checkpoint=checkpoint)), name)

    def test_unknown_name_and_custom_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'choose small, medium, or gpt2'):
            select_model_config('large')
        custom = GPTConfig(n_layer=1, n_head=1, n_embd=8)
        self.assertEqual(select_model_config(checkpoint={'config': asdict(custom)}), custom)
        self.assertEqual(model_size_name(custom), 'custom')


if __name__ == '__main__':
    unittest.main()
