# English movie-subtitle data for a small GPT-2

Research date: 2026-08-19

## Bottom line

There is enough *raw* English subtitle text to train a small GPT-2-style model from scratch. The strongest published baseline is **OpenSubtitles2018 English: 447,000 subtitle files, 140,000 distinct IMDb IDs, 441 million sentences, and 3.2 billion OPUS-tokenized tokens**. The IMDb IDs represent movies **or TV episodes**, and the counts are not unique GPT-2 BPE tokens. ([OpenSubtitles2018 paper, Table 1](https://aclanthology.org/L18-1275.pdf))

The live OPUS API reports **1,274,672 English XML documents, 1,136,279,357 sentence fragments/alignment units, and 8,482,331,872 source tokens** in OpenSubtitles v2024. For comparison, its exact v2018 English row is **446,612 documents, 441,450,451 fragments, and 3,235,391,790 tokens**. The v2024 number should be treated as an upper bound before cleaning: OPUS explicitly says its token totals include duplicate sentences and documents, and the current release includes alignments among alternative uploads. ([v2024 API result](https://opus.nlpl.eu/opusapi/?source=en&target=de&corpus=OpenSubtitles&version=v2024&preprocessing=xml), [v2018 API result](https://opus.nlpl.eu/opusapi/?source=en&corpus=OpenSubtitles&version=v2018&preprocessing=xml), [OPUS dataset page](https://opus.nlpl.eu/datasets/OpenSubtitles))

The main constraint is therefore not raw volume. It is obtaining enough **clean, unique, English, movie-only, legally usable** text after title-level splitting, language/origin filtering, and deduplication. Publicly downloadable does not mean permissively licensed.

My practical recommendation is:

1. Prototype the parser, tokenizer, and evaluation on Cornell Movie-Dialogs and/or MovieSum.
2. If noncommercial research is acceptable, use OPUS OpenSubtitles XML for volume, but filter and deduplicate aggressively before tokenization.
3. Establish a fine-tuned GPT-2 Small baseline first. For a from-scratch run, start near **34M parameters** (8 layers, width 512, 8 heads, 512-token context, and a 16K domain tokenizer) on **600–800M cleaned tokens measured with that tokenizer**. Only move to the official 124M-size GPT-2 after measuring the genuinely usable corpus.
4. If the intended product is commercial or its weights will be redistributed, get a legal review or limit training to text with affirmative rights clearance.

## Dataset candidates

| Dataset | What is available | Format and language | License/access reality | Best role |
|---|---|---|---|---|
| [OpenSubtitles via OPUS](https://opus.nlpl.eu/datasets/OpenSubtitles) | v2018 English has 447K subtitle files, 140K movie-or-TV IMDb IDs, 441M sentences, and 3.2B OPUS tokens. The current multilingual page reports 94 languages overall. ([v2018 paper](https://aclanthology.org/L18-1275.pdf)) | Each subtitle is represented as XML with tokenized sentences, timestamps, language/source metadata, and movie/TV metadata; the original source is usually SRT. ([format description](https://aclanthology.org/L18-1275.pdf)) | OPUS says it does not own the text and only believes it can redistribute it. OpenSubtitles' current terms prohibit commercial use. No conventional permissive content license is shown on the OPUS corpus page. ([OPUS disclaimer](https://opus.nlpl.eu/datasets/OpenSubtitles), [OpenSubtitles terms](https://opensubtitles.tawk.help/article/terms-of-service)) | Only candidate here with enough volume for from-scratch pretraining; treat as research/noncommercial unless rights are cleared. |
| [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) | 304,713 utterances, 220,579 exchanges, 9,035 characters, and 617 movies. ConvoKit reconstructs 83,097 conversations. ([Cornell](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), [ConvoKit](https://convokit.cornell.edu/documentation/movie.html)) | English delimited text files with utterance, conversation, character, title, genre, year, ratings, and raw-source URL metadata; ConvoKit provides a JSON-oriented conversation representation. ([ConvoKit schema](https://convokit.cornell.edu/documentation/movie.html), [official archive](https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip)) | Publicly downloadable, but neither the official page nor the archive README states a dataset-content license. The README says it was extracted from publicly available movie scripts. | Excellent pipeline fixture, structured-dialogue supplement, and held-out evaluation set; much too small for a 124M model from scratch. |
| [MovieSum](https://github.com/saxenarohit/MovieSum) | 2,200 screenplays with Wikipedia summaries; 1,800/200/200 train/validation/test split; mean reported screenplay length 34,275. | English XML with scene, stage-direction, scene-description, character, and dialogue tags, plus IMDb IDs. | Dataset declares CC BY-NC 4.0, so commercial use is not allowed. Because the records contain third-party screenplays, the declared dataset license should not be assumed to settle every underlying right; that is a legal-risk inference, not a claim made by the authors. ([repository and license](https://github.com/saxenarohit/MovieSum)) | Strongest structured screenplay source with an explicit license; good for format/structure adaptation, not subtitle volume. |
| [Film Corpus 2.0](https://nlds.soe.ucsc.edu/node/34) | 1,068 complete scripts; 960 have dialogue separated from scene descriptions. | English TXT, genre-grouped; scraped from IMSDb. | Form-gated research download with no broad content license stated on the official page. IMSDb describes scripts as educational-use material. ([UCSC page](https://nlds.soe.ucsc.edu/node/34), [IMSDb disclaimer](https://imsdb.com/disclaimer.html)) | Useful screenplay/dialogue supplement for research, after rights review. |
| [ScriptBase](https://github.com/EdinburghNLP/scriptbase) | 1,276 movies in ScriptBase-alpha; 917 in ScriptBase-J. | Tar archives with original HTML/plain text; the 917-film subset adds manually corrected text and automatically annotated XML, plus metadata and summaries. | The public repository does not state a content license. It contains material crawled from screenplay sites, IMDb, Wikipedia, and Jinni. ([repository data description](https://github.com/EdinburghNLP/scriptbase)) | Research reference and screenplay-structure source; not a cleanly licensed training foundation. |
| [MovieNet](https://movienet.github.io/) | 815 subtitle files (84.4MB unpacked) and 479 script files (101.8MB unpacked) from its 1,100-movie collection. | Subtitle, script, synopsis, JSON metadata, and multimodal annotations. | Requires registration and acceptance of a user-service agreement; this is controlled research access, not an unambiguous permissive license. ([official download page](https://movienet.github.io/)) | Helpful if movie-only identity and multimodal metadata matter more than volume. |
| [SPOLIN](https://github.com/wise-east/spolin) | Latest release has 112,597 dialogue pairs: 35,736 from Cornell, 59,815 from SubTle, and the rest from a podcast, including constructed negatives. | English JSON dialogue pairs and ConvoKit/ParlAI/Hugging Face adapters. | CC BY-NC 4.0. ([official repository](https://github.com/wise-east/spolin)) | Small pair-level fine-tuning/evaluation set. It also ships code and links to fine-tuned GPT-2 weights. |
| [Gutenberg Dialogue](https://github.com/ricsinaruto/gutenberg-dialog) | 14,773,741 English utterances in 2,526,877 dialogues; average utterance length 22.17 words. | English plus six other languages; dialogue files and a reproducible extraction pipeline. | The project declares MIT and its paper says copyrighted books were removed before extraction. ([repository/license](https://github.com/ricsinaruto/gutenberg-dialog), [paper](https://aclanthology.org/2021.eacl-main.11.pdf)) | Legally cleaner dialogue pretraining/supplement, but it is literary rather than movie language. |

### Useful OpenSubtitles preparation code

PolyAI's open-source preparation pipeline converts OPUS OpenSubtitles into context-response examples in sharded TFRecord or JSON. Its English run produced 316,891,717 examples. The authors warn that adjacent subtitle lines are not guaranteed to come from different speakers. ([pipeline README](https://github.com/PolyAI-LDN/conversational-datasets/blob/master/opensubtitles/README.md))

That warning is important: subtitle blocks are segmented for screen timing, not for dialogue modeling. A plain "line A -> line B" training transform silently introduces false turns.

## How many tokens are available?

### Counts supported directly by sources

| Source | Count | What the count means |
|---|---:|---|
| OpenSubtitles2018 English | **3,235,391,790** | Moses/Kytea-style OPUS tokens across 446,612 documents and 441,450,451 sentence fragments, with duplicates; not GPT-2 BPE. ([API](https://opus.nlpl.eu/opusapi/?source=en&corpus=OpenSubtitles&version=v2018&preprocessing=xml), [paper](https://aclanthology.org/L18-1275.pdf)) |
| OpenSubtitles v2024 English | **8,482,331,872** | Current API `source_tokens` across 1,274,672 documents and 1,136,279,357 fragments before deduplication. ([API](https://opus.nlpl.eu/opusapi/?source=en&target=de&corpus=OpenSubtitles&version=v2024&preprocessing=xml)) |
| Cornell Movie-Dialogs, ConvoKit | **4,479,102 text-only / 4,783,547 newline-delimited GPT-2 BPE tokens** | Reproducible local counts described below. |
| MovieSum | **~75.4M reported length units** | 2,200 × mean screenplay length 34,275. The repository does not identify this number as GPT-2 BPE, so it should not be called a GPT-2 token count. ([repository](https://github.com/saxenarohit/MovieSum)) |

The OpenSubtitles2018 paper explains that its tokenization uses Moses scripts (or Kytea for Chinese), while GPT-2 uses byte-level BPE. The two units are not interchangeable. ([OpenSubtitles preprocessing](https://aclanthology.org/L18-1275.pdf), [official GPT-2 code/model repository](https://github.com/openai/gpt-2))

### Reproducible Cornell GPT-2 count

I downloaded Cornell's official [ConvoKit version](https://convokit.cornell.edu/documentation/movie.html), selected its 304,446 nonempty utterances, and encoded the text without special tokens using OpenAI's official `encoder.json` and `vocab.bpe`. OpenAI's download script identifies those tokenizer files and their official location. ([OpenAI download script](https://github.com/openai/gpt-2/blob/master/download_model.py))

Results:

```text
nonempty utterances                       304,446
GPT-2 BPE tokens, utterance text only   4,479,102
GPT-2 BPE tokens, one newline each      4,783,547
```

The two totals show why serialization must be specified: a newline separator contributes about one token per utterance. These are useful exact counts for Cornell's ConvoKit release, but its BPE behavior should not be projected blindly across all of OpenSubtitles: caption markup, OCR noise, names, translations, and language-identification errors can change tokenization materially. The older raw archive reports 304,713 utterance records, so counts also vary slightly by official release and empty-text handling. ([original Cornell release](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), [ConvoKit release](https://convokit.cornell.edu/documentation/movie.html))

### What model size fits the data?

The Chinchilla study found that compute-optimal model size and training-token count scale together; its 70B-parameter model used 1.4T tokens, implying the often-used planning approximation of about **20 training tokens per parameter**. This is a rough compute-allocation heuristic, not a minimum-data law and not specific to subtitles. ([primary paper](https://arxiv.org/abs/2203.15556))

| Model parameters | Rough compute-optimal token target |
|---:|---:|
| 10M | 200M |
| 30M | 600M |
| 34M | 680M |
| 50M | 1.0B |
| 60M | 1.2B |
| GPT-2 Small, 124M | 2.48B |

OpenAI now describes the corrected small GPT-2 parameter count as 124M. ([official repository note](https://github.com/openai/gpt-2/blob/master/README.md))

Therefore:

- Cornell alone supplies only 0.039 GPT-2 tokens per parameter for a 124M model; it is suitable for fine-tuning or evaluation, not from-scratch pretraining.
- OpenSubtitles2018 has enough raw volume to reach the approximate 2.48B target for 124M, but it is not yet proven that 2.48B *clean, unique, rights-usable GPT-2 BPE tokens* remain.
- A roughly 34M model is a safer first from-scratch experiment because 600–800M cleaned tokens bracket the approximate 680M compute-optimal point before committing to the full 124M run.

## Has this been done before?

Yes. There is no single canonical checkpoint I found that exactly matches "GPT-2 Small trained from scratch only on cleaned English movie subtitles," but several close precedents answer the engineering question.

1. **GPT-2 trained on OpenSubtitles for dialogue.** The Gutenberg Dialogue study adapted the pretrained 117M GPT-2 to Gutenberg and OpenSubtitles dialogue. It used three previous utterances as history, batch size 2, and reports that the English OpenSubtitles run reached its validation minimum after one epoch, about two days on one RTX 2080 Ti. The paper compares GPT-2 models across seven languages and found the less noisy Gutenberg dialogue often transferred better than OpenSubtitles. ([paper, Sections 5.2–5.3](https://aclanthology.org/2021.eacl-main.11.pdf), [released code/data/models](https://github.com/ricsinaruto/gutenberg-dialog))

2. **GPT-2 fine-tuned on cleaned English OpenSubtitles.** A 2022 LREC study cleaned and title-deduplicated an OpenSubtitles dialogue benchmark to roughly 1.0M training context-response examples, then fine-tuned a 12-layer, 768-hidden, 12-head GPT-2. It showed that 34.49% of the original OpenSubtitles test samples were identical to training samples and that overlap greatly inflated BLEU, making movie/title-level deduplication non-optional. ([paper and Tables 3–4](https://aclanthology.org/2022.lrec-1.16.pdf))

3. **Fine-tuned GPT-2 released for movie-derived pairs.** SPOLIN derives tens of thousands of pairs from Cornell and SubTle, publishes GPT-2 training code, and links fine-tuned GPT-2 weights. ([official repository](https://github.com/wise-east/spolin))

4. **Genre-controlled screenplay generation.** A public project fine-tuned GPT-2 and BART on IMSDb screenplays using genre tokens and releases training/evaluation code and processed-data links. Its repository license covers the project code; it should not be read as a license grant from screenplay owners. ([project repository](https://github.com/shahhaard47/Script-Generation), [IMSDb disclaimer](https://imsdb.com/disclaimer.html))

5. **Long script generation with GPT-2-medium.** A theatre-script project fine-tuned GPT-2-medium on 12,641 scripts: 1,067 movies, 6,057 TV episodes, and 5,517 theatre plays. The authors report character-consistency and synopsis-drift problems and say copyright/licensing prevented release of most theatre data. ([paper](https://aclanthology.org/2022.wnu-1.4.pdf))

6. **A smaller academic screenplay baseline.** A Stanford project fine-tuned GPT-2 for one and three epochs on about 1,300 IMSDb screenplays (reported as about 39M words). Three epochs improved screenplay formatting, but the authors still found much dialogue incoherent or contradictory. ([report](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report035.pdf), [code](https://github.com/antle1/cs224n_FinalProject_ProGeT))

The combined lesson is consistent: domain adaptation quickly teaches screenplay/subtitle *surface form*, while character consistency, long-range plot coherence, false dialogue turns, duplicated titles, and licensing remain the hard parts.

## Data-quality and legal issues to design around

### 1. Split by title before generating examples

OpenSubtitles contains alternative uploads and versions of the same work. A published study found 34.49% exact train/test overlap in a commonly used OpenSubtitles dialogue split. Keep all files for an IMDb title in one split, then deduplicate within and across titles, before deriving context-response windows. ([overlap study](https://aclanthology.org/2022.lrec-1.16.pdf))

### 2. Movie-only and originally-English are separate filters

The v2018 English row covers 140K distinct IMDb IDs, explicitly defined as movies **or TV episodes**. An English subtitle can also be a translation of non-English audio. OPUS XML includes the original language and IMDb-linked metadata needed for filtering, but the paper does not claim every English subtitle is originally English. ([dataset and metadata description](https://aclanthology.org/L18-1275.pdf))

### 3. Captions are not speaker turns

Subtitles ordinarily lack reliable character labels. The PolyAI preprocessing authors explicitly warn that adjacent lines need not be different speakers. Train a language model on document sequences if the goal is next-token modeling; only build dialogue pairs where speaker changes can be inferred with high confidence. ([pipeline warning](https://github.com/PolyAI-LDN/conversational-datasets/blob/master/opensubtitles/README.md))

### 4. Remove non-dialogue and high-risk noise

Useful filters include hearing-impaired sound cues, music/lyrics, uploader advertisements, URLs, OCR garbage, repeated credits, formatting tags, extremely short/long captions, machine-translated uploads, and near-duplicate alternative subtitles. OPUS metadata identifies hearing-impaired and machine-translated uploads and records quality-related fields. ([OpenSubtitles2018 preprocessing](https://aclanthology.org/L18-1275.pdf))

### 5. Treat download permission, code license, dataset license, and underlying copyright separately

OPUS says it does not own the subtitle text; OpenSubtitles prohibits commercial use; Cornell and ScriptBase do not state a dataset-content license on their official release pages; MovieSum and SPOLIN explicitly limit use to noncommercial purposes. ([OPUS](https://opus.nlpl.eu/datasets/OpenSubtitles), [OpenSubtitles terms](https://opensubtitles.tawk.help/article/terms-of-service), [Cornell](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), [ScriptBase](https://github.com/EdinburghNLP/scriptbase), [MovieSum](https://github.com/saxenarohit/MovieSum), [SPOLIN](https://github.com/wise-east/spolin))

For a private research experiment, this risk profile may be acceptable to the project owner. For a public/commercial model or redistributed cleaned corpus, it needs an explicit rights strategy and legal review.

## Suggested next experiment

1. Download a small title-bounded sample of OPUS XML plus Cornell.
2. Preserve document/IMDb boundaries and add explicit `<|movie|>`, `<|caption|>`, and `<|endoftext|>` separators; do not fabricate speaker IDs.
3. Filter to English text from originally-English titles, and movie-only when IMDb title type is available.
4. Normalize markup, remove high-risk noise, and run exact plus MinHash/semantic near-deduplication at caption, subtitle-file, and title levels.
5. Freeze train/validation/test by IMDb title.
6. Tokenize with the tokenizer the target model will actually use (OpenAI GPT-2 BPE for the fine-tuning baseline; the proposed 16K domain BPE for the scratch model) and publish a data card containing raw tokens, post-filter tokens, unique-token estimate, title counts, duplicate-removal rates, and license/source buckets.
7. Fine-tune pretrained GPT-2 Small first to establish a low-cost domain-adaptation baseline.
8. Then train an approximately 34M-parameter GPT-2-style model (8 layers, width 512, 8 heads, 16K vocabulary, 512-token context) on 600–800M cleaned tokens. Compare it with the fine-tuned baseline on held-out perplexity, repetition, memorized-quote probes, dialogue coherence, and character/name consistency.

The immediate go/no-go metric is not the 8.48B raw OPUS number. It is the exact post-filter token count under the target tokenizer, together with a source/license distribution that matches the intended use.
