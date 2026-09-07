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
