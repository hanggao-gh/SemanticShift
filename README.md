# Semantic Shift

This repository contains code for the **Semantic Shift** project: measuring how embedding geometry changes as text is concatenated / chunked, and how that impacts retrieval and downstream consistency.

> Folder layout (current)
- `src/` — core research code (metrics, splitters, Qdrant utilities)
- `data/` — place raw datasets / cached corpora
- `results/` — outputs (JSON/CSV/plots)

## Quick start

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Notes
- Some experiments depend on GPU / `torch`.
- Some scripts use **Qdrant** (local or remote).
- Some scripts use **OpenAI embeddings** (requires `OPENAI_API_KEY`).

### 2) Run a script

Most scripts are written to be executed from the project root, with the `src/` package available.

## What’s inside

### Core metrics
- **MPD (Mean Pairwise Distance)**: average cosine distance among a set/window of embeddings.
- **Semantic Shift**: combines local drift (adjacent distances) with global dispersion (MPD).

See:
- `src/mean_pairwise_distance.py`
- `src/mean_semantic_shift.py`

### Cpair Cmean Correlation
Theorem 1, Semantic Dilution.

See:
- `src/cpair_cmean_correlation.py`

### Qdrant preprocessing
Preprocess text into sentences, embed, and upsert to Qdrant.

See:
- `src/qdrant_store.py`
- `src/preprocess_to_qdrant.py`

### Retrieval geometry / anisotropy & consistency
Utilities for analyzing embedding space geometry and retrieval consistency, optionally backed by Qdrant.

See:
- `src/anisotropy_retrieval_consistency.py`
- `src/tools.py`

### Splitters (chunking)
Implements and evaluates multiple chunkers/splitters (fixed sentence groups, LlamaIndex semantic splitter, and a semantic-shift drift splitter).

See:
- `src/semantic_shift_splitter.py`


## Configuration & credentials

Environment variables commonly used:
- `OPENAI_API_KEY` — required if you use OpenAI embeddings.
- `QDRANT_URL`, `QDRANT_API_KEY` — for hosted Qdrant (optional).
- `HF_ENDPOINT` — some scripts set a HF mirror; you may remove/override it if not needed.

## Reproducibility tips

- Fix random seeds where applicable.
- Log model name, dataset name, and key hyperparameters into `results/`.

## License

MIT — see [LICENSE](LICENSE).
