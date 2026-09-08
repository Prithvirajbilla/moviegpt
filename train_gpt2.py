import os
import math
import time
import inspect
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import random
import shutil
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


# All presets retain GPT-2 token IDs and a 1024-token context. The vocabulary is
# padded to 50304 for matrix multiplication; these are parameter-size presets.
MODEL_PRESETS = {
    "small": dict(n_layer=6, n_head=6, n_embd=384),
    "medium": dict(n_layer=8, n_head=8, n_embd=512),
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
}


def preset_config(name):
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown MOVIEGPT_MODEL_SIZE {name!r}; choose small, medium, or gpt2")
    return GPTConfig(vocab_size=50304, **MODEL_PRESETS[name])


def select_model_config(name=None, checkpoint=None):
    requested = preset_config(name) if name is not None else None
    if checkpoint is not None:
        saved = GPTConfig(**checkpoint["config"])
        if requested is not None and requested != saved:
            raise ValueError("MOVIEGPT_MODEL_SIZE conflicts with the checkpoint architecture. "
                             "Omit it to use the saved model, or start a new run for a different size.")
        return saved
    return requested or preset_config("gpt2")


def model_size_name(config):
    return next((name for name in MODEL_PRESETS if preset_config(name) == config), "custom")


def model_locations(name):
    # Preserve the existing GPT-2 paths and isolate the smaller experiments.
    suffix = f"-{name}" if name in {"small", "medium"} else ""
    return (f"log/{name}" if suffix else "log", f"relentlessml/moviegpt{suffix}")

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
from huggingface_hub import HfApi, hf_hub_download, get_token
from huggingface_hub.utils import EntryNotFoundError
from moviegpt_hub import export_model, upload_export
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, IterableDataset


class MovieDocuments:
    """Download/cache one Parquet shard at a time; never use HF datasets."""

    def __init__(self, repo_id, revision=None):
        info = HfApi().dataset_info(repo_id, revision=revision)
        self.repo_id = repo_id
        self.revision = info.sha  # pin both splits to the same snapshot
        self.files = [item.rfilename for item in info.siblings]

    def texts(self, split):
        for text, _ in self.records(split):
            yield text

    def records(self, split, cursor=None):
        """Seek to a shard/row without rereading or retokenizing preceding shards."""
        files = sorted(f for f in self.files
                       if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
        if not files:
            raise ValueError(f"No Parquet files for {split} in {self.repo_id}")
        cursor = cursor or {"shard": 0, "row": 0}
        for shard_index in range(cursor["shard"], len(files)):
            path = hf_hub_download(self.repo_id, files[shard_index], repo_type="dataset",
                                   revision=self.revision)
            start_row = cursor["row"] if shard_index == cursor["shard"] else 0
            with pq.ParquetFile(path) as parquet:
                row_offset = 0
                for group in range(parquet.num_row_groups):
                    count = parquet.metadata.row_group(group).num_rows
                    if row_offset + count <= start_row:
                        row_offset += count
                        continue
                    for batch in parquet.iter_batches(batch_size=64, columns=["text"], row_groups=[group]):
                        for text in batch.column(0).to_pylist():
                            row_offset += 1
                            if row_offset <= start_row:
                                continue
                            if isinstance(text, str) and text.strip():
                                yield text, {"shard": shard_index, "row": row_offset}


class PackedMovieBatches(IterableDataset):
    """Pack next-token batches; ranks take disjoint blocks of one token stream."""

    def __init__(self, documents, tokenizer, B, T, rank, world_size, split):
        self.documents, self.tokenizer = documents, tokenizer
        self.B, self.T = B, T
        self.rank, self.world_size, self.split = rank, world_size, split
        self.reset()

    def reset(self):
        self.cursor = {"shard": 0, "row": 0}
        self.pending = []
        self.next_block = 0

    def state_dict(self):
        return {"cursor": dict(self.cursor), "pending": list(self.pending),
                "next_block": self.next_block, "B": self.B, "T": self.T,
                "rank": self.rank, "world_size": self.world_size, "split": self.split}

    def load_state_dict(self, state):
        for name in ("B", "T", "rank", "world_size", "split"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Cannot resume loader with different {name}")
        self.cursor = dict(state["cursor"])
        self.pending = list(state["pending"])
        self.next_block = state["next_block"]

    def __iter__(self):
        size = self.B * self.T
        records = iter(self.documents.records(self.split, self.cursor))
        while True:
            while len(self.pending) < size + 1:
                try:
                    text, self.cursor = next(records)
                except StopIteration:
                    return  # drop the incomplete final batch
                ids = self.tokenizer.encode(text, add_special_tokens=False,
                                            truncation=False, verbose=False)
                if ids and ids[-1] != self.tokenizer.eos_token_id:
                    ids.append(self.tokenizer.eos_token_id)
                self.pending.extend(ids)
            block = self.next_block
            self.next_block += 1
            tokens = self.pending[:size + 1] if block % self.world_size == self.rank else None
            self.pending = self.pending[size:]
            # Advance state before yielding so a checkpoint points to the NEXT batch.
            if tokens is not None:
                tokens = torch.tensor(tokens, dtype=torch.long)
                yield tokens[:-1].view(self.B, self.T), tokens[1:].view(self.B, self.T)


class MovieDataLoader:
    def __init__(self, documents, tokenizer, B, T, process_rank, num_processes, split):
        self.B, self.T = B, T
        self.dataset = PackedMovieBatches(documents, tokenizer, B, T,
                                         process_rank, num_processes, split)
        # A separate generator keeps iterator construction out of training RNG state.
        self.loader = DataLoader(self.dataset, batch_size=None, num_workers=0,
                                 generator=torch.Generator().manual_seed(1337))
        self.reset()

    def reset(self):
        self.epoch = 0
        self.dataset.reset()
        self.iterator = iter(self.loader)

    def state_dict(self):
        return {"epoch": self.epoch, "dataset": self.dataset.state_dict()}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.dataset.load_state_dict(state["dataset"])
        self.iterator = iter(self.loader)

    def next_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.dataset.reset()
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise ValueError("Split has too few tokens for this batch size and rank") from exc


# Authored diagnostic examples, not a calibrated benchmark or training data.
# The correct completion is first; scoring uses only completion tokens.
COMPREHENSION_CHECKS = [
    ("I put the keys in the drawer. Where are the keys?\n", ["In the drawer.", "On the roof.", "In the ocean."]),
    ("Maya handed the ticket to Leo. Who has the ticket now?\n", ["Leo has it.", "Maya still has it.", "Nobody ever had a ticket."]),
    ("The door was locked, so I used my key. Now I can\n", ["open the door.", "eat the door.", "turn into a key."]),
    ("Are you hungry?\nYes, I haven't eaten all day.\nThen let's\n", ["get some dinner.", "throw away all the food.", "paint the moon."]),
    ("It started raining. I don't want to get wet. I'll take\n", ["an umbrella.", "a fork.", "a photograph of sand."]),
    ("Nina missed the last train. She needed another way home, so she\n", ["called a taxi.", "boarded the train she missed.", "became a railway station."]),
    ("I promised not to tell anyone your secret. You can\n", ["trust me to keep it private.", "expect me to announce it to everyone.", "drink the secret for breakfast."]),
    ("The glass fell off the table and shattered. Please be careful of\n", ["the broken glass on the floor.", "the intact glass on the table.", "the flying refrigerator."]),
]


def evaluate_comprehension(model, tokenizer, device):
    correct = 0
    with torch.no_grad():
        for prompt, choices in COMPREHENSION_CHECKS:
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            rows = [prefix + tokenizer.encode(c, add_special_tokens=False) for c in choices]
            length = max(map(len, rows))
            tokens = torch.full((len(rows), length), tokenizer.eos_token_id,
                                dtype=torch.long, device=device)
            mask = torch.zeros_like(tokens)
            for i, row in enumerate(rows):
                tokens[i, :len(row)] = torch.tensor(row, device=device)
                mask[i, len(prefix):len(row)] = 1
            logits, _ = model(tokens)
            correct += get_most_likely_row(tokens, mask, logits) == 0
    return correct / len(COMPREHENSION_CHECKS)

# -----------------------------------------------------------------------------
# Score completion likelihood for the comprehension diagnostic
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each candidate completion
    # the one with the lowest loss should be the most likely
    best = avg_loss.min()
    if (avg_loss == best).sum().item() != 1:
        return -1  # ties provide no evidence of comprehension
    return avg_loss.argmin().item()

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def capture_rng(device):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if str(device).startswith("cuda"):
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif str(device) == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def read_checkpoint(source, log_dir):
    if source == "latest":
        paths = list(Path(log_dir).glob("model_*.pt"))
        paths = [p for p in paths if p.stem.removeprefix("model_").isdigit()]
        if not paths:
            raise FileNotFoundError(f"No model checkpoints in {log_dir}")
        source = max(paths, key=lambda p: int(p.stem.removeprefix("model_")))
    elif str(source).startswith("hf://"):
        repo_id = str(source)[5:]
        revision = HfApi().model_info(repo_id, revision=os.environ.get("MOVIEGPT_RESUME_REVISION")).sha
        try:
            source = hf_hub_download(repo_id, "training_state.pt", repo_type="model", revision=revision)
        except EntryNotFoundError:
            # Previous uploads contained only Transformers weights + step metadata.
            import json
            from transformers import GPT2LMHeadModel
            metadata_path = hf_hub_download(repo_id, "training_metadata.json", repo_type="model", revision=revision)
            metadata = json.loads(Path(metadata_path).read_text())
            pretrained = GPT2LMHeadModel.from_pretrained(repo_id, revision=revision)
            config = pretrained.config
            weights = {}
            for name, value in pretrained.state_dict().items():
                if name.endswith((".attn.bias", ".attn.masked_bias")):
                    continue
                if name.endswith(("attn.c_attn.weight", "attn.c_proj.weight",
                                  "mlp.c_fc.weight", "mlp.c_proj.weight")):
                    value = value.t().contiguous()
                weights[name] = value
            return {"model": weights, "config": asdict(GPTConfig(
                        block_size=config.n_positions, vocab_size=config.vocab_size,
                        n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd)),
                    "step": metadata["optimizer_steps_completed"],
                    "val_loss": metadata["validation_loss"],
                    "training": {"dataset_id": metadata["dataset"],
                                 "dataset_revision": metadata["dataset_revision"]}}
    # Legacy scripts pickled this one known dataclass under __main__ or train_gpt2.
    # Keep restricted loading enabled; never fall back to arbitrary pickle execution.
    with torch.serialization.safe_globals([(GPTConfig, "__main__.GPTConfig"),
                                            (GPTConfig, "train_gpt2.GPTConfig")]):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") not in (None, 2):
        raise ValueError("Unsupported checkpoint version")
    if not isinstance(checkpoint.get("step"), int) or checkpoint["step"] < 0:
        raise ValueError("Checkpoint must contain a nonnegative completed step count")
    if isinstance(checkpoint["config"], GPTConfig):
        checkpoint["config"] = asdict(checkpoint["config"])
    return checkpoint


def save_checkpoint(path, model, optimizer, step, val_loss, rank_states, training, target_steps):
    checkpoint = {
        "checkpoint_version": 2, "model": model.state_dict(), "config": asdict(model.config),
        "optimizer": optimizer.state_dict(), "step": step, "val_loss": val_loss,
        "rank_states": rank_states, "training": training, "target_steps": target_steps,
    }
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)  # interrupted writes cannot replace the previous checkpoint


def restore_training(checkpoint, model, optimizer, loader, rank, device, training):
    if checkpoint.get("checkpoint_version") == 2:
        if checkpoint["training"] != training:
            changed = sorted(k for k in set(training) | set(checkpoint["training"])
                             if training.get(k) != checkpoint["training"].get(k))
            raise ValueError(f"Resume configuration changed: {', '.join(changed)}. "
                             "Use the saved dataset, tokenizer, batch, device type and GPU count.")
        if len(checkpoint["rank_states"]) != training["world_size"]:
            raise ValueError("Checkpoint has incomplete per-rank state")
    model.load_state_dict(checkpoint["model"])
    if checkpoint.get("checkpoint_version") == 2:
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["rank_states"][rank]
        loader.load_state_dict(state["loader"])
        restore_rng(state["rng"], device)  # restore AFTER constructing model and iterators
    else:
        warnings.warn("Legacy weights-only checkpoint: restoring model and step, but AdamW "
                      "state, dataset revision and data position were not saved. Optimizer "
                      "and data start fresh; this is not an exact training resume.")
    return checkpoint["step"]


def prepare_logs(log_dir, resume_step=None):
    """Retain valid history; archive any abandoned records newer than the checkpoint."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "log.txt"
    if resume_step is None:
        path.write_text("")
    elif path.exists():
        lines = path.read_text().splitlines(keepends=True)
        kept = []
        for line in lines:
            fields = line.split()
            if len(fields) == 3 and fields[0].isdigit():
                step = int(fields[0])
                if step < resume_step or (step == resume_step and fields[1] != "train"):
                    kept.append(line)
        if kept != lines:
            shutil.copyfile(path, log_dir / f"log.before-resume-{time.time_ns()}.txt")
            path.write_text("".join(kept))
    else:
        path.touch()
    return str(path)


def main():
    global master_process
    requested_size = os.environ.get("MOVIEGPT_MODEL_SIZE")
    if requested_size is not None:
        preset_config(requested_size)  # validate before any downloads
    default_log_dir, _ = model_locations(requested_size or "gpt2")
    log_dir = os.environ.get("MOVIEGPT_LOG_DIR", default_log_dir)
    resume_source = os.environ.get("MOVIEGPT_RESUME")
    checkpoint = read_checkpoint(resume_source, log_dir) if resume_source else None
    saved_training = checkpoint.get("training", {}) if checkpoint else {}
    model_config = select_model_config(requested_size, checkpoint)
    model_size = model_size_name(model_config)
    default_log_dir, default_repo = model_locations(model_size)
    log_dir = os.environ.get("MOVIEGPT_LOG_DIR", default_log_dir)
    hub_repo = os.environ.get("MOVIEGPT_MODEL_REPO", default_repo)
    hub_upload = os.environ.get("MOVIEGPT_HUB_UPLOAD", "1")
    hub_private = os.environ.get("MOVIEGPT_MODEL_PRIVATE", "0")
    if hub_upload not in {"0", "1"} or hub_private not in {"0", "1"}:
        raise ValueError("MOVIEGPT_HUB_UPLOAD and MOVIEGPT_MODEL_PRIVATE must be 0 or 1")
    hub_upload, hub_private = hub_upload == "1", hub_private == "1"
    if hub_upload and not get_token():
        raise RuntimeError("Hugging Face uploads require a write-capable HF_TOKEN or Hub login. "
                           "Set MOVIEGPT_HUB_UPLOAD=0 for local training and export only.")
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = AutoTokenizer.from_pretrained("gpt2")
    documents = MovieDocuments(
        os.environ.get("MOVIEGPT_DATASET", saved_training.get("dataset_id", "relentlessml/moviegpt")),
        os.environ.get("MOVIEGPT_REVISION", saved_training.get("dataset_revision")),
    )

    total_batch_size = saved_training.get("total_batch_size", 524288)
    B = int(os.environ.get("MOVIEGPT_MICRO_BATCH", saved_training.get("B", 64)))
    T = saved_training.get("T", 1024)
    if B <= 0:
        raise ValueError("MOVIEGPT_MICRO_BATCH must be positive")
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = MovieDataLoader(documents=documents, tokenizer=enc, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = MovieDataLoader(documents=documents, tokenizer=enc, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="validation")

    torch.set_float32_matmul_precision('high')

    # create model
    model = GPT(model_config)
    if master_process:
        parameter_count = sum(p.numel() for p in model.parameters())
        print(f"model size: {model_size} | parameters: {parameter_count:,}")
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)
    use_compile = False
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    max_lr = saved_training.get("max_lr", 6e-4)
    min_lr = saved_training.get("min_lr", max_lr * 0.1)
    warmup_steps = saved_training.get("warmup_steps", 715)
    max_steps = int(os.environ.get("MOVIEGPT_MAX_STEPS",
                                  checkpoint.get("target_steps", 19073*2) if checkpoint else 19073*2))
    decay_steps = saved_training.get("decay_steps", max_steps)
    if max_steps < 1:
        raise ValueError("MOVIEGPT_MAX_STEPS must be positive")
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it >= decay_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (decay_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    training = {
        "B": B, "T": T, "world_size": ddp_world_size, "total_batch_size": total_batch_size,
        "dataset_id": documents.repo_id, "dataset_revision": documents.revision,
        "tokenizer_sha256": hashlib.sha256(enc.backend_tokenizer.to_str().encode()).hexdigest(),
        "max_lr": max_lr, "min_lr": min_lr, "warmup_steps": warmup_steps, "decay_steps": decay_steps,
        "device_type": device.split(":")[0],
    }
    start_step = 0
    if checkpoint:
        start_step = restore_training(checkpoint, raw_model, optimizer, train_loader,
                                      ddp_rank, device, training)
        if start_step > max_steps:
            raise ValueError("Checkpoint step exceeds MOVIEGPT_MAX_STEPS (total target, not additional steps)")
        if master_process:
            print(f"Resuming after {start_step} completed optimizer updates")
    if master_process:
        log_file = prepare_logs(log_dir, start_step if checkpoint else None)
    else:
        log_file = os.path.join(log_dir, "log.txt")
    val_loss_accum = torch.tensor(checkpoint["val_loss"] if checkpoint else float("nan"), device=device)
    resuming = checkpoint is not None
    del checkpoint  # release CPU copies of model and optimizer state

    for step in range(start_step, max_steps + 1):
        t0 = time.time()
        last_step = (step == max_steps) # evaluate/save after the final optimizer update

        # Checkpoints were saved after evaluation; do not repeat it on resume.
        resumed_boundary = resuming and step == start_step

        # once in a while evaluate our validation loss
        if (step % 250 == 0 or last_step) and not resumed_boundary:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f} | perplexity: {math.exp(min(val_loss_accum.item(), 80)):.2f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                    f.write(f"{step} perplexity {math.exp(min(val_loss_accum.item(), 80)):.4f}\n")

        # Small context/comprehension diagnostic; all ranks run the same fixed set.
        if (step % 250 == 0 or last_step) and not resumed_boundary:
            model.eval()
            accuracy = evaluate_comprehension(raw_model, enc, device)
            if master_process:
                print(f"comprehension sanity accuracy: {accuracy:.3f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} comprehension_sanity {accuracy:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile) and not resumed_boundary:
            model.eval()
            num_return_sequences = 4
            max_length = 96
            tokens = enc.encode("Where were you last night?\nI was at the station.\n", add_special_tokens=False)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :len(enc)] # exclude padded vocabulary entries
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")
                if master_process:
                    with open(os.path.join(log_dir, "samples.txt"), "a") as f:
                        f.write(f"step {step} sample {i}: {decoded}\n\n")

        if step > 0 and (step % 5000 == 0 or last_step) and not resumed_boundary:
            local_state = {"loader": train_loader.state_dict(), "rng": capture_rng(device)}
            rank_states = [None] * ddp_world_size if master_process else None
            if ddp:
                dist.gather_object(local_state, rank_states, dst=0)
            else:
                rank_states = [local_state]
            if master_process:
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                save_checkpoint(checkpoint_path, raw_model, optimizer, step,
                                val_loss_accum.item(), rank_states, training, max_steps)
                export_path = export_model(
                    raw_model, enc, os.path.join(log_dir, "huggingface", f"step-{step:05d}"),
                    step=step, dataset_id=documents.repo_id, dataset_revision=documents.revision,
                    val_loss=val_loss_accum.item(), log_file=log_file, training_state=checkpoint_path,
                    model_size=model_size,
                )
                if hub_upload:
                    try:
                        result = upload_export(export_path, hub_repo, private=hub_private)
                        print(f"Uploaded model: {result}")
                    except Exception as exc:
                        # Keep training on transient Hub/auth failures; the export can be retried.
                        print(f"Model upload failed ({type(exc).__name__}); local export: {export_path}. "
                              f"Retry: python moviegpt_hub.py {export_path} --repo-id {hub_repo}"
                                  + (" --public" if not hub_private else ""))


        if last_step:
            break

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
