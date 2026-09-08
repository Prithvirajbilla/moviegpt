from dataclasses import asdict
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from train_gpt2 import (GPT, GPTConfig, MovieDataLoader, MovieDocuments, capture_rng,
                        read_checkpoint, restore_training, save_checkpoint, prepare_logs, preset_config)
from test_training_data import Documents, Tokenizer


class ResumeTests(unittest.TestCase):
    def make_loader(self, rank=0, world=1):
        return MovieDataLoader(Documents(), Tokenizer(), 1, 3, rank, world, 'train')

    def test_loader_resumes_next_batch_for_each_rank_across_epochs(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                loader = self.make_loader(rank, 2)
                loader.next_batch()
                state = loader.state_dict()
                expected = [loader.next_batch() for _ in range(7)]
                restored = self.make_loader(rank, 2)
                restored.load_state_dict(state)
                for x, y in expected:
                    actual_x, actual_y = restored.next_batch()
                    self.assertTrue(torch.equal(x, actual_x))
                    self.assertTrue(torch.equal(y, actual_y))
                self.assertEqual(restored.epoch, loader.epoch)

    def test_records_seek_to_saved_shard_and_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shard.parquet'
            pq.write_table(pa.table({'text': ['old', None, 'resume here', 'last']}), path, row_group_size=2)
            documents = MovieDocuments.__new__(MovieDocuments)
            documents.repo_id, documents.revision = 'owner/repo', 'fixed'
            documents.files = ['data/train-a.parquet', 'data/train-b.parquet']
            with patch('train_gpt2.hf_hub_download', return_value=str(path)) as download:
                records = list(documents.records('train', {'shard': 1, 'row': 2}))
            self.assertEqual(records, [('resume here', {'shard': 1, 'row': 3}),
                                       ('last', {'shard': 1, 'row': 4})])
            download.assert_called_once_with('owner/repo', 'data/train-b.parquet',
                                             repo_type='dataset', revision='fixed')

    def test_resume_matches_uninterrupted_optimizer_update_and_rng(self):
        torch.manual_seed(7)
        random.seed(7)
        config = GPTConfig(vocab_size=100, block_size=3, n_layer=1, n_head=1, n_embd=8)
        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loader = self.make_loader()
        training = {'world_size': 1, 'B': 1, 'dataset_revision': 'fixed'}

        def update(model, optimizer, loader):
            x, y = loader.next_batch()
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            return loss.detach()

        update(model, optimizer, loader)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_00001.pt'
            save_checkpoint(path, model, optimizer, 1, 2.5,
                            [{'loader': loader.state_dict(), 'rng': capture_rng('cpu')}], training, 10)
            expected_random = (random.random(), torch.rand(4))
            expected_loss = update(model, optimizer, loader)
            expected_weights = {key: value.clone() for key, value in model.state_dict().items()}
            saved = read_checkpoint('latest', directory)
            restored = GPT(GPTConfig(**saved['config']))
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
            restored_loader = self.make_loader()
            self.assertEqual(restore_training(saved, restored, restored_optimizer,
                                              restored_loader, 0, 'cpu', training), 1)
            self.assertEqual(random.random(), expected_random[0])
            torch.testing.assert_close(torch.rand(4), expected_random[1], rtol=0, atol=0)
            actual_loss = update(restored, restored_optimizer, restored_loader)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                restore_training(saved, restored, restored_optimizer, restored_loader,
                                 0, 'cpu', {**training, 'B': 2})

    def test_legacy_checkpoint_restores_weights_and_step_with_warning(self):
        model = GPT(GPTConfig(vocab_size=100, block_size=3, n_layer=1, n_head=1, n_embd=8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_05000.pt'
            torch.save({'model': model.state_dict(), 'config': model.config, 'step': 5000,
                        'val_loss': 3.0}, path)
            saved = read_checkpoint(path, directory)
            restored = GPT(GPTConfig(**saved['config']))
            with patch('train_gpt2.warnings.warn') as warning:
                step = restore_training(saved, restored, torch.optim.AdamW(restored.parameters()),
                                        self.make_loader(), 0, 'cpu', {'world_size': 1})
            self.assertIn("Legacy weights-only", warning.call_args.args[0])
            self.assertEqual(step, 5000)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)

    def test_resume_archives_abandoned_log_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log.txt'
            original = '0 train 4.0\n1 val 3.0\n1 train 3.5\n2 train 2.5\n'
            path.write_text(original)
            prepare_logs(directory, 1)
            self.assertEqual(path.read_text(), '0 train 4.0\n1 val 3.0\n')
            backups = list(Path(directory).glob('log.before-resume-*.txt'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_00001.pt'
            path.write_bytes(b'previous checkpoint')
            model = GPT(GPTConfig(vocab_size=100, block_size=3, n_layer=1, n_head=1, n_embd=8))
            with patch('train_gpt2.torch.save', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    save_checkpoint(Path(directory) / 'model_00002.pt', model, torch.optim.AdamW(model.parameters()),
                                    2, 2.0, [], {}, 10)
            self.assertEqual(path.read_bytes(), b'previous checkpoint')



    def checkpoint_fixture(self, directory, filename, size, step):
        path = Path(directory) / filename
        torch.save({'model': {'probe': torch.tensor([1])}, 'config': asdict(preset_config(size)),
                    'step': step, 'val_loss': 3.0}, path)
        return path

    def test_latest_filters_architecture_and_uses_saved_step(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = self.checkpoint_fixture(directory, 'model_00001.pt', 'small', 100)
            self.checkpoint_fixture(directory, 'model_small_99999.pt', 'small', 50)
            self.checkpoint_fixture(directory, 'model_medium_20000.pt', 'medium', 20000)
            saved = read_checkpoint('auto', directory, model_size='small')
            self.assertEqual(saved['step'], 100)
            self.assertEqual(saved['_resume_path'], str(expected))

    def test_corrupt_newest_falls_back_but_never_starts_fresh_over_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self.checkpoint_fixture(directory, 'model_00001.pt', 'small', 1)
            corrupt = Path(directory) / 'model_small_00002.pt'
            corrupt.write_bytes(b'incomplete checkpoint')
            with patch('train_gpt2.warnings.warn') as warning:
                self.assertEqual(read_checkpoint('auto', directory, 'small')['step'], 1)
                warning.assert_called_once()
                valid.unlink()
                with self.assertRaises(FileNotFoundError):
                    read_checkpoint('auto', directory, 'small')
            self.assertEqual(corrupt.read_bytes(), b'incomplete checkpoint')

    def test_legacy_shared_directory_is_searched_for_requested_size(self):
        with tempfile.TemporaryDirectory() as directory:
            small_dir = Path(directory) / 'small'
            self.checkpoint_fixture(directory, 'model_00500.pt', 'medium', 500)
            self.assertIsNone(read_checkpoint('auto', small_dir, 'small', (directory,)))
            self.checkpoint_fixture(directory, 'model_00100.pt', 'small', 100)
            self.assertEqual(read_checkpoint('auto', small_dir, 'small', (directory,))['step'], 100)

    def test_fresh_run_and_wrong_size_do_not_clobber_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(read_checkpoint('auto', directory, 'small'))
            self.checkpoint_fixture(directory, 'model_00500.pt', 'medium', 500)
            with self.assertRaises(FileNotFoundError):
                read_checkpoint('auto', directory, 'small')
            with self.assertRaises(FileExistsError):
                read_checkpoint('none', directory)
            with self.assertRaises(FileExistsError):
                prepare_logs(directory)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'log.txt'
            log.write_text('0 train 4.0\n')
            with self.assertRaises(FileExistsError):
                read_checkpoint('auto', directory, 'small')
            self.assertEqual(log.read_text(), '0 train 4.0\n')

    def test_checkpoint_cannot_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_small_00001.pt'
            path.write_bytes(b'previous checkpoint')
            model = GPT(GPTConfig(vocab_size=100, block_size=3, n_layer=1, n_head=1, n_embd=8))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 1, 2.0, [], {}, 10)
            self.assertEqual(path.read_bytes(), b'previous checkpoint')


if __name__ == '__main__':
    unittest.main()
