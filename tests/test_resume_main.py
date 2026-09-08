import os
import io
from contextlib import redirect_stdout
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import train_gpt2 as training
import unittest


class ResumeMainTests(unittest.TestCase):
    def test_main_continues_at_next_update_and_exports_full_state(self):
        # Tiny, offline end-to-end exercise of main(): checkpoint after update 1, resume to 2.
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        backend = Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, **{f'w{i}': i+2 for i in range(98)}}, unk_token='<unk>'))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', bos_token='<eos>', eos_token='<eos>')
        class Docs:
            repo_id = 'test/fixture'
            revision = 'pinned'
            def records(self, split, cursor=None):
                for i in range((cursor or {}).get('row', 0), 20):
                    yield 'w0 w1 w2 w3 ' * 40, {'shard': 0, 'row': i + 1}

        docs = Docs()
        model = training.GPT(training.GPTConfig(vocab_size=100, block_size=128, n_layer=1, n_head=1, n_embd=8))
        training.master_process = False
        optimizer = model.configure_optimizers(weight_decay=.1, learning_rate=.001, device_type='cpu')
        loader = training.MovieDataLoader(docs, tokenizer, 1, 8, 0, 1, 'train')
        x,y = loader.next_batch()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, loss = model(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        settings = {'B':1,'T':8,'world_size':1,'total_batch_size':8,
                    'dataset_id':docs.repo_id,'dataset_revision':docs.revision,
                    'tokenizer_sha256':hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                    'max_lr':.001,'min_lr':.0001,'warmup_steps':1,'decay_steps':4,'device_type':'cpu'}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'model_00001.pt'
            training.save_checkpoint(checkpoint, model, optimizer, 1, loss.item(),
                                     [{'loader':loader.state_dict(),'rng':training.capture_rng('cpu')}], settings, 4)
            expected_next = loader.next_batch()
            optimizer.zero_grad()
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                _, loss = model(*expected_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            env = {'MOVIEGPT_RESUME':str(checkpoint), 'MOVIEGPT_MAX_STEPS':'2', 'MOVIEGPT_LOG_DIR':directory,
                   'MOVIEGPT_HUB_UPLOAD':'0', 'RANK':'-1'}
            with patch.dict(os.environ, env), patch.object(training.AutoTokenizer,'from_pretrained',return_value=tokenizer), \
                 patch.object(training,'MovieDocuments',return_value=docs), patch.object(torch.cuda,'is_available',return_value=False), \
                 patch.object(torch.backends.mps,'is_available',return_value=False):
                with redirect_stdout(io.StringIO()):
                    training.main()
            saved = training.read_checkpoint(Path(directory)/'model_custom_00002.pt', directory)
            assert saved['step']==2
            assert saved['rank_states'][0]['loader']==loader.state_dict()
            assert all(state['step'].item()==2 for state in saved['optimizer']['state'].values())
            for name,tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, saved['model'][name], rtol=0, atol=0)
            assert (Path(directory)/'huggingface/step-00002/training_state.pt').is_file()
