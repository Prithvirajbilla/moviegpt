import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from train_gpt2 import (MovieDocuments, MovieDataLoader, PackedMovieBatches,
                        GPT, GPTConfig, get_most_likely_row)


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, **kwargs):
        return [int(x) for x in text.split()]


class Documents:
    def records(self, split, cursor=None):
        texts = ['1 2 3 99', '4 5 6', '7 8 9 10 11 12 13 14']
        for index in range((cursor or {}).get('row', 0), len(texts)):
            yield texts[index], {'shard': 0, 'row': index + 1}


class TrainingDataTests(unittest.TestCase):
    def test_packing_shift_eos_and_rank_partition(self):
        all_batches = list(PackedMovieBatches(Documents(), Tokenizer(), 1, 3, 0, 1, 'train'))
        stream = [1, 2, 3, 99, 4, 5, 6, 99, 7, 8, 9, 10, 11, 12, 13, 14, 99]
        for i, (x, y) in enumerate(all_batches):
            self.assertEqual(x.flatten().tolist(), stream[i*3:i*3+3])
            self.assertEqual(y.flatten().tolist(), stream[i*3+1:i*3+4])
        for rank in range(2):
            batches = list(PackedMovieBatches(Documents(), Tokenizer(), 1, 3, rank, 2, 'train'))
            self.assertEqual(len(batches), len(all_batches[rank::2]))
            for actual, expected in zip(batches, all_batches[rank::2]):
                self.assertTrue(torch.equal(actual[0], expected[0]))

    def test_reset_and_epoch_rollover(self):
        loader = MovieDataLoader(Documents(), Tokenizer(), 1, 3, 0, 1, 'train')
        first = loader.next_batch()
        for _ in range(4):
            loader.next_batch()
        self.assertTrue(torch.equal(first[0], loader.next_batch()[0]))
        loader.reset()
        self.assertTrue(torch.equal(first[0], loader.next_batch()[0]))

    def test_too_small_split_fails_clearly(self):
        loader = MovieDataLoader(Documents(), Tokenizer(), 64, 1024, 0, 1, 'train')
        with self.assertRaisesRegex(ValueError, 'too few tokens'):
            loader.next_batch()

    def test_parquet_split_selection_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.parquet'
            pq.write_table(pa.table({'text': ['hello', None, '', 'world']}), path)
            documents = MovieDocuments.__new__(MovieDocuments)
            documents.repo_id, documents.revision = 'owner/repo', 'fixed-sha'
            documents.files = ['data/train-a.parquet', 'data/validation-a.parquet', 'data/test-a.parquet']
            with patch('train_gpt2.hf_hub_download', return_value=str(path)) as download:
                self.assertEqual(list(documents.texts('validation')), ['hello', 'world'])
                download.assert_called_once_with('owner/repo', 'data/validation-a.parquet',
                                                 repo_type='dataset', revision='fixed-sha')
            with self.assertRaisesRegex(ValueError, 'No Parquet'):
                list(documents.texts('missing'))

    def test_completion_mask_excludes_prompt_and_padding(self):
        tokens = torch.tensor([[1, 2, 3, 0], [1, 2, 4, 4]])
        mask = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 1]])
        logits = torch.zeros(2, 4, 5)
        logits[0, 1, 3] = 8
        logits[0, 2, 0] = -100  # padding cannot affect the score
        self.assertEqual(get_most_likely_row(tokens, mask, logits), 0)

    def test_tied_scores_do_not_count_as_correct(self):
        tokens = torch.tensor([[1, 2], [1, 3]])
        mask = torch.tensor([[0, 1], [0, 1]])
        self.assertEqual(get_most_likely_row(tokens, mask, torch.zeros(2, 2, 5)), -1)

    def test_tiny_model_training_batch(self):
        x, y = MovieDataLoader(Documents(), Tokenizer(), 1, 3, 0, 1, 'train').next_batch()
        model = GPT(GPTConfig(block_size=3, vocab_size=100, n_layer=1, n_head=1, n_embd=8))
        _, loss = model(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
