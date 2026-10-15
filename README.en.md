# glycanPUG Workspace Overview

> 🌐 中文版：[README.md](./README.md)

The purpose of each folder under this repository root:

| Folder | Description | README |
|--------|-------------|--------|
| `MASK/` | **Main project**: PUG (called W0 in the code) glycan-structure GNN pre-training and downstream tasks. Based on the best checkpoint at epoch **95** | [README](./MASK/README.en.md) |
| `MASK_i/` | **Ablation study** version, same structure as MASK, based on the best checkpoint at epoch **124**, only ran the taxall task | [README](./MASK_i/README.en.md) |
| `GT/` | **Comparison experiment**: extracts embeddings with GlycanGT for a fair comparison with PUG | [README](./GT/README.en.md) |
| `GlycanPUG/` | Organized **official version for GitHub upload** | [README](./GlycanPUG/README.md) |
| `downstream/` | **Raw data and build scripts for downstream tasks** (source of each task's CSVs, see table below) | — |

## Inside downstream/

| Subfolder | Description |
|-----------|-------------|
| `tax(ML)/` | Taxonomy data (**clean version, no uncertain structures**, sourced from GlycanML); `分拆.py` splits it by level for `MASK/95/95轮tax/` and the GT comparison |
| `taxall/` | Taxonomy "all-taxa" data (contains **uncertain structures** that GlycanGT cannot process), 8 levels from superkingdom → species |
| `imme/` | Immunogenicity data construction (data sourced from GlycanML, cleaned/merged into `imm_stata.csv`) |
| `linkpred/` | GlyConnect linking-type data construction (crawl/process, produces `glyconnect_structures.csv`) |
| `dic.csv` | The **original full** glycan dictionary downloaded from GlyTouCan; the `dic.csv` inside `MASK/` and `MASK_i/` is the version after `deldownstata.py` removed leakage rows |

## Naming note

The model was originally called **W0** and later renamed **PUG**; the code still uses the W0 naming (e.g., `glycanW0_classifier.py`, `get_embeddings_W0.py`). For the code-vs-paper naming of graph components and features, see [MASK/README.en.md](./MASK/README.en.md).
