# MovieGPT

An experiment in pretraining a language model entirely on movie text: what happens when everything a model has ever read comes from the movies?

The ambition is a potentially **brain rot LLM** with cinematic instincts: unnecessary plot twists, dramatic confessions, suspicious basements, and a comeback for everything. These are hypotheses to investigate, not demonstrated capabilities. The project asks how a narrow training diet shapes language, assumptions, and predictions about what happens next.

## Current status

Local raw data, processed data, generated dataset packages, and test files have been removed. This repository retains preparation scripts, research notes, and this project plan. There is no implemented model-training pipeline or trained checkpoint here.

The existing `scripts/` are historical corpus utilities. They include a Friends cleaner and a publisher for mixed movie/TV data, so they need adaptation for this experiment. Their default input files are no longer present. `requirements.txt` covers the earlier data pipeline, not a complete training environment.

## What movie-only pretraining means

- Initialize every model weight randomly. Loading a pretrained GPT-2 checkpoint would carry over its original training data.
- Train the tokenizer exclusively on the movie training split, without externally learned vocabulary or merges.
- Begin with verified English movie subtitles or dialogue transcripts. Treat full screenplays as a separate experiment because stage directions add information absent from subtitles.
- Exclude television, including Friends, as well as reviews, plot summaries, Wikipedia, general web text, instruction datasets, and synthetic LLM dialogue.
- Keep the primary checkpoint free of later instruction tuning or preference training. Label any adapted checkpoint separately.

The first version is a text-only base model that continues unfinished dialogue. Instruction-following chat behavior is a separate research question.

## Prepare the corpus

1. Collect movie text with documented provenance and permitted training use. Record source, title identifier, language, original language, version, and usage terms. Dataset permissions are separate from the repository's code license.
2. Verify movie identity using title metadata. Reject television and quarantine records whose media type cannot be established. Earlier mixed subtitle exports do not establish movie-only provenance.
3. Clean encoding, timestamps, markup, advertisements, and duplicate captions. Preserve punctuation and dialogue order. Document whether sound cues remain; do not invent speakers or scene boundaries.
4. Group alternate subtitle releases and near-duplicate scripts before a deterministic 90/5/5 train/validation/test split **by movie**. All versions of a title belong in one partition. Consider grouping franchises for a harder holdout. Exclude benchmark source titles before training either the tokenizer or model.
5. Train the tokenizer on the training partition, then encode each partition separately. Preserve document boundaries with an end-of-document token. Use separate sequences or masked cross-document attention to avoid learning continuations between unrelated films. Mask padding out of the loss.
6. Record unique titles, retained tokens, exclusions, hashes, split assignments, and cleaning decisions. Count unique corpus tokens separately from tokens revisited across epochs.

Keep future raw corpora, token shards, and checkpoints outside version control.

## Train from scratch

Start with a small decoder-only Transformer to validate the experiment. This is a proposed pilot, not a measured optimum:

| Setting | Initial experiment |
| --- | --- |
| Architecture | GPT-style causal decoder, random initialization |
| Size | 12 layers, width 768, 12 attention heads; roughly 110M parameters with tied embeddings and a 32K vocabulary |
| Tokenizer | Byte-level BPE trained on movie training text, approximately 32K tokens |
| Context | 1,024 tokens |
| Objective | Next-token cross-entropy |
| Optimizer | AdamW, learning-rate warmup and decay, gradient clipping |
| Precision | BF16 where supported; select batch size after a memory pilot |
| Checkpoints | Model, tokenizer, optimizer, scheduler, RNG state, and data position |

Choose the training-token budget after measuring corpus size and pilot throughput. Scale model size and training data together; repeating a small corpus does not create more unique material. The [compute-optimal training study](https://arxiv.org/abs/2203.15556) provides background for this tradeoff, but does not establish an optimum for movie text.

Implementation sequence:

1. Add a tokenizer trainer, document-aware data loader, training entry point, evaluation entry point, and versioned run configuration.
2. Instantiate the model from configuration without loading pretrained weights. Hugging Face's [causal language modeling example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) supports training from scratch and is an implementation reference.
3. Run a short pilot to check decreasing loss, finite gradients, correct padding masks, checkpoint resume, and held-out validation loss.
4. Record effective batch size in tokens, learning rate, seeds, corpus hash, tokens processed, throughput, and peak memory. Estimate runtime from measured tokens per second before scaling up.
5. Train against the declared budget, select checkpoints using validation results, and evaluate the untouched test partition once choices are fixed.

## Measure the brain rot

Define a fixed prompt set and scoring rules before comparing checkpoints. The question is whether movie-only pretraining produces dramatic habits alongside useful dialogue coherence.

| Hypothesis | Evaluation |
| --- | --- |
| Every basement noise means danger | Score dramatic versus mundane continuations across matched prompts |
| Everyone has a comeback writer | Blind human ratings of wit, relevance, and unnecessary escalation |
| Strangers are destined for romance | Compare romantic and ordinary continuations of neutral meetings |
| Every accusation deserves a monologue | Compare elaborate confessions with short, context-appropriate responses |
| Characters can hold a conversation | Held-out dialogue loss, next-response ranking, and speaker consistency |
| It does more than quote karaoke | Compare seen-title and held-out-title prompts; search generations for matching training spans |

Track repetition, incoherence, and general-text performance too. “Brain rot” is a playful label for measurable tendencies, not an explanation for every bad generation. Subtitles alone do not provide the visual content of a film.

Optional controls are separate models trained on general text and a movie/general mixture. Match architecture, compute, decoding, and evaluation treatment. Keep the main MovieGPT weights and tokenizer movie-only. A shared fixed byte tokenizer can avoid introducing an externally learned vocabulary; if tokenizers differ, compare common task metrics or bits per byte instead of raw token perplexity. Keep evaluation text out of training and report variability across seeds.

## Next milestones

1. Establish a verifiable movie-only corpus and title-level holdouts.
2. Implement tokenizer training and a reproducible small pretraining run.
3. Run dialogue, trope, and memorization evaluations.
4. Produce a model card with provenance, training budget, results, and examples of coherent dialogue and characteristic failures.

Earlier research remains in `docs/research/`. It includes broader movie/TV ideas; the stricter movie-only definition above governs this experiment.

## Current training script

Run `pip install -r requirements.txt`, then `python train_gpt2.py` (or
`torchrun --standalone --nproc_per_node=8 train_gpt2.py` for eight CUDA GPUs).
The existing defaults are a large training run: microbatch 64, context 1024,
and 524,288 tokens per optimizer step. Set `MOVIEGPT_MICRO_BATCH=4` for a small
memory pilot; new checkpoints remember the batch size for resume.

`train_gpt2.py` downloads `relentlessml/moviegpt` through `huggingface_hub`,
reads Parquet in batches with PyArrow, tokenizes with Transformers'
`AutoTokenizer`, and feeds packed batches through PyTorch `DataLoader`.
It does not require the Hugging Face `datasets` package or local FineWeb shards.
Transformers supplies the tokenizer; the Hub client supplies dataset downloads.
Shards are downloaded on demand into the Hugging Face cache, so the first visit
to each shard requires network access and disk space. Each rank tokenizes the same
stream and takes disjoint blocks; this saves corpus-sized RAM at the cost of
repeated tokenization across ranks. Documents retain one trailing EOS separator;
packed contexts can cross document boundaries. Incomplete final batches are dropped.

Set `MOVIEGPT_DATASET` to override the repository and `MOVIEGPT_REVISION` to pin
a commit. Both splits use the same resolved revision. Existing Hub login or
`HF_TOKEN` works for private repositories. The loader uses the published `train`
and `validation` splits and leaves `test` untouched. These are document holdouts,
not guaranteed movie/title holdouts: the dataset includes Friends and subtitle
chunks whose original movie boundaries are unavailable.

Every 250 steps the script records validation loss and perplexity over a fixed
20-batch validation window, plus an eight-item authored comprehension sanity
check (chance is roughly 1/3). This replaces HellaSwag, which measures commonsense
ending selection rather than directly rating generated sentences. The diagnostic
scores completion tokens only and tests simple context tracking and plausible
responses; it is too small to serve as a general comprehension benchmark.
Perplexity uses the capped exponent of loss (cap 80 to avoid overflow).

Fixed-seed dialogue continuations are saved to `log/samples.txt`. Compare samples
between checkpoints for grammatical sentences, relevant replies, consistent facts,
and repetition. Neither perplexity nor the sanity score proves coherence; review
these samples as well. Validation currently samples a fixed prefix, not the whole
validation corpus. The script retains the existing GPT-2 tokenizer and randomly
initialized model; training a movie-only tokenizer remains a separate milestone.

Run loader and scoring checks with `python -m unittest discover -s tests -v`.

Open `example.ipynb` to monitor training loss (raw and smoothed), validation loss,
perplexity, comprehension accuracy, and generated dialogue. Install its optional
dependencies with `python -m pip install -r requirements-notebook.txt`, select the
project's Python kernel, and run all cells. Rerun to refresh from `log/log.txt`;
set `LOG_DIR` for another run or `SAVE_PLOTS = True` to export PNGs.

## Upload trained models to Hugging Face

Checkpoint exports upload automatically to the **public model repository**
`relentlessml/moviegpt`, separate from the dataset repository. Authenticate with
`hf auth login` using a token that can write to that model repository, or provide
`HF_TOKEN` in the training environment. Do not put the token in the notebook or
source files. Training checks for a token before loading the dataset or model.

Every 5,000 completed optimizer updates, and after the final update, rank zero
saves the local `.pt` checkpoint and a Transformers export under
`log/huggingface/step-NNNNN/`. The export contains Safetensors weights, model and
generation configurations, the tokenizer, a model card, training metadata with
the pinned dataset revision, and the metrics log. It loads with
`AutoModelForCausalLM.from_pretrained("relentlessml/moviegpt")` and
`AutoTokenizer.from_pretrained("relentlessml/moviegpt")` once uploaded.

Each upload commits the latest export to the repository root; earlier exports
remain accessible through Hub commit history. The original training model's
Linear matrices are transposed into Transformers GPT-2's Conv1D layout, and
padded vocabulary IDs are suppressed during generation. New exports include `training_state.pt` with optimizer, schedule, RNG and data
position for training resume; older exports contain model weights only.
The dataset itself is not uploaded into the model repository.

Configuration:

- `MOVIEGPT_MODEL_REPO`: destination model repository (default `relentlessml/moviegpt`).
- `MOVIEGPT_MODEL_PRIVATE=1`: create a private repository instead of public.
  This setting only controls creation; existing repository visibility is unchanged.
- `MOVIEGPT_HUB_UPLOAD=0`: train and export locally without uploading or requiring a token.

If an upload fails, training continues and prints the saved export path. Retry it
without training again:

```bash
python moviegpt_hub.py log/huggingface/step-19073 --repo-id relentlessml/moviegpt --public
```

Uploads run synchronously on rank zero at checkpoint boundaries; allow time for
exporting and transferring the model. Both local checkpoints and exports consume
disk space. No trained checkpoint is included in this source repository.

## Run training from Colab

Use a GPU runtime and run the repository's script as a separate Python process.
The training script needs the companion `moviegpt_hub.py`; copying only the
training script into a notebook cell is insufficient. The companion has a
project-specific name to avoid the unrelated PyPI package `model_hub`.

In a fresh Colab runtime (use the PR branch until it is merged):

```python
!git clone --branch codex/huggingface-training-and-monitoring https://github.com/Prithvirajbilla/moviegpt.git /content/moviegpt
%cd /content/moviegpt
%pip install -r requirements.txt
```

Log in using a write-capable Hugging Face token through the interactive prompt:

```python
from huggingface_hub import login
login()
```

Before launching, reduce `B = 64` in `train_gpt2.py` for the GPU's memory; start
with `B = 4` for a pilot and adjust after measuring memory use. The existing script
uses BF16 on CUDA, so choose a BF16-capable GPU for this configuration.
Then launch from the repository directory:

```python
!python train_gpt2.py
```

If `/content/moviegpt` already exists, use that checkout and pull its latest branch
instead of cloning again. `example.ipynb` is the log-plotting notebook; training is
launched separately. Setting `MOVIEGPT_HUB_UPLOAD=0` before launch skips login and
uploads, while retaining local exports.

## Continue a saved training run

Resume the most recent local checkpoint:

```bash
MOVIEGPT_RESUME=latest python train_gpt2.py
```

Or choose a specific checkpoint:

```bash
MOVIEGPT_RESUME=log/model_05000.pt python train_gpt2.py
```

Resume from the model repository on another machine:

```bash
MOVIEGPT_RESUME=hf://relentlessml/moviegpt python train_gpt2.py
```

Use `MOVIEGPT_RESUME_REVISION` to choose a particular model-repository commit.
Hub reads are pinned to one commit so model weights and metadata cannot come from
different uploads. This model revision is separate from the saved dataset revision.

New checkpoints restore weights, AdamW state, completed update count, learning-rate
schedule, per-rank Python/PyTorch RNG, dataset revision, and each rank's next data
position. The loader saves the Parquet shard/row and leftover tokens, skips earlier
shards/row groups, and continues with the next batch instead of replaying the corpus.
Epoch rollover is preserved. Checkpoints are written atomically every 5,000 completed
updates and at the end; a crash can still lose work after the latest saved checkpoint.
An already-running copy of the old script will continue writing the old checkpoint
format until you relaunch with this version.

The default total budget is **38,146 updates**. A full-state resume uses the saved
total target by default. `MOVIEGPT_MAX_STEPS=50000` changes the total stopping point,
not the number of additional updates. Extending a run preserves its original cosine
schedule; after the saved decay endpoint, learning rate stays at its saved minimum.
Resuming a checkpoint that already meets the target performs no further updates.

Keep the saved microbatch, sequence length, GPU count, effective batch size, tokenizer,
and dataset revision for a full-state resume; incompatible changes fail explicitly.
Hardware/software differences and nondeterministic kernels can still change numerical
results even with restored RNG. Choose your memory settings before the initial run.

For old local `.pt` checkpoints and old Hub exports without `training_state.pt`, the
script restores the learned weights and saved step, then warns that AdamW and the
data position start fresh. Old Hub metadata also supplies the dataset revision.
There is no way to recover optimizer/RNG/data state that was never saved. Set
`MOVIEGPT_MICRO_BATCH` explicitly for those old checkpoints, since they did not save it.
This is weights-only continuation, not an exact replay of the interrupted run.

Set `MOVIEGPT_LOG_DIR` to choose the local checkpoint/log directory. Resume retains
history through the checkpoint; if existing metrics extend past it, their complete
original log is archived before the abandoned tail is removed. Generation samples
remain appended in `samples.txt`. On a new machine, earlier logs stay available in
the Hub repository; the local log starts from the resumed step.

New Hub exports include `training_state.pt`, so they take more storage and upload
time than inference-only exports. Local exports use a hard link for that file when
possible to avoid another disk copy. `AutoModelForCausalLM` still loads the separate
Safetensors weights for inference.
