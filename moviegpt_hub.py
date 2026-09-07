"""Export MovieGPT to Transformers and publish a saved export to Hugging Face."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import GPT2Config, GPT2LMHeadModel


@torch.no_grad()
def export_model(model, tokenizer, folder, *, step, dataset_id, dataset_revision,
                 val_loss, log_file=None):
    """Copy weights to CPU, converting Linear matrices to GPT-2 Conv1D layout."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = model.config
    hf_config = GPT2Config(
        vocab_size=config.vocab_size, n_positions=config.block_size,
        n_ctx=config.block_size, n_layer=config.n_layer, n_head=config.n_head,
        n_embd=config.n_embd, activation_function="gelu_new",
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        layer_norm_epsilon=1e-5, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )
    exported = GPT2LMHeadModel(hf_config)
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight",
                  "mlp.c_fc.weight", "mlp.c_proj.weight")
    source = model.state_dict()
    for name, target in exported.state_dict().items():
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            continue  # generated causal buffers in older Transformers versions
        value = source[name].detach().cpu()
        if name.endswith(transposed):
            value = value.t()
        if value.shape != target.shape:
            raise ValueError(f"Cannot export {name}: {value.shape} != {target.shape}")
        target.copy_(value)
    exported.eval()
    # Training pads the vocabulary to 50304 for efficient matrix multiplication.
    # Keep those weights for loss fidelity but never generate invalid token IDs.
    exported.generation_config.suppress_tokens = list(range(len(tokenizer), config.vocab_size)) or None
    exported.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    metadata = {
        "optimizer_steps_completed": step,
        "validation_loss": val_loss,
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "tokenizer": tokenizer.name_or_path,
        "initialization": "random",
        "validation_scope": "fixed 20-batch window",
    }
    (folder / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if log_file is not None and Path(log_file).is_file():
        shutil.copyfile(log_file, folder / "log.txt")
    (folder / "README.md").write_text(f'''---
library_name: transformers
pipeline_tag: text-generation
language:
- en
datasets:
- {dataset_id}
tags:
- gpt2
- moviegpt
---

# MovieGPT

A GPT-2 architecture language model trained from random weights on movie and TV
dialogue using the GPT-2 tokenizer. This is a base language model, not an
instruction-tuned chat model.

- Completed optimizer updates: {step}
- Validation cross-entropy: {val_loss:.4f} nats/token
- Dataset: [{dataset_id}](https://huggingface.co/datasets/{dataset_id})
- Dataset revision: `{dataset_revision}`

Validation measures a fixed 20-batch window, not the full corpus. Low loss does
not establish comprehension or coherent dialogue. The corpus includes Friends,
Cornell Movie-Dialogs, and OpenSubtitles chunks; document splits do not guarantee
title-level isolation. Generated text may repeat training dialogue or be incoherent.
Consult the dataset card for source provenance and licensing; no new model license
is asserted here.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "YOUR_MODEL_REPOSITORY"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)
inputs = tokenizer("Where were you last night?\\n", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=80, do_sample=True, top_k=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The repository contains Safetensors weights, configuration, tokenizer files,
`training_metadata.json`, and a metrics log when available. Optimizer state and
data-loader position are not included; this export is for inference, not exact
training resume. Padded vocabulary IDs are suppressed by the generation config.
''')
    return folder


def upload_export(folder, repo_id, *, private=True):
    """Upload only the model export, never the surrounding workspace or dataset."""
    folder = Path(folder)
    required = ("config.json", "tokenizer_config.json", "training_metadata.json", "README.md")
    if any(not (folder / name).is_file() for name in required) or not list(folder.glob("*.safetensors")):
        raise ValueError(f"Incomplete model export: {folder}")
    metadata = json.loads((folder / "training_metadata.json").read_text())
    api = HfApi()
    # private controls newly created repositories; never change existing visibility.
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.txt", "tokenizer.model",
                        "README.md", "log.txt"],
        commit_message=f"MovieGPT after {metadata['optimizer_steps_completed']} optimizer updates",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload or retry a saved MovieGPT export")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true", help="Create a public repository (default: private)")
    args = parser.parse_args()
    print(upload_export(args.folder, args.repo_id, private=not args.public))
