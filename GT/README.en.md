# GT: GlycanGT Comparison Experiment

This folder is for **comparison experiments**, using [GlycanGT](https://huggingface.co/) (a TokenGT-based glycan graph model) to extract embeddings for a fair comparison with this project's PUG model (called W0 in the code).

> 🌐 中文版：[README.md](./README.md)

> **Naming note**: The model was originally called **W0** and later renamed **PUG**; the code still uses the W0 naming. The `glycangt_classifier.py` here follows the same pipeline as PUG's `glycanW0_classifier.py`, differing only in the data source.

## Directory Structure

```
GT/
├── get_embeddings.py          # Extract embeddings using GlycanGT
└── glycangt_classifier.py     # Classification training (same as PUG's glycanW0_classifier.py, seed 42 only)
```

## Usage

### 1. Set up the GlycanGT environment

GlycanGT is quite new. This project does **not** use the package versions specified in the GlycanGT paper; instead it runs GlycanGT with this project's own dependency configuration (verified to work). After cloning the GlycanGT repository, run in its root directory:

```bash
pip install .
```

### 2. Extract embeddings

Place `get_embeddings.py` and `split_files/` (WURCS CSVs split by taxonomic level, from PUG's `95轮tax/`; note: **do not use `taxall/`** — it contains uncertain structures that GlycanGT cannot process) together into the **GlycanGT repository root**, then run:

```bash
# Reads ./split_files by default, outputs to ./split_files_processed
py -3.13 get_embeddings.py --mode split_files
```

This produces `split_files_processed/` (the "processed" folder) containing CSVs with an appended embedding column.

### 3. Classification training

`glycangt_classifier.py` uses the exact same training pipeline as PUG's `glycanW0_classifier.py` (CB Focal Loss, AdamW, MultiStageReduceLROnPlateau, EMA early stopping, per-epoch threshold optimization) for a fair comparison. To avoid duplication, **only the seed-42 code is included** here:

```bash
py -3.13 glycangt_classifier.py --work_dir ./split_files_processed
```
