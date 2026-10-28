# glycanPUG: Glycan-Structure GNN Pre-training and Downstream Tasks

> 🌐 中文版：[README.md](./README.md)

This repository parses glycan WURCS strings into heterogeneous graphs (big/small nodes + big/small edges), pre-trains a heterogeneous GAT with MLM (masked language modeling), extracts 1027-dim graph-level embeddings, and trains classification heads for downstream tasks such as taxonomy, glycan-linking type, and immunogenicity. The model was originally called **W0** and later renamed **PUG**. The purpose of each folder under the repository root:

| Folder | Description | README |
|--------|-------------|--------|
| `MASK/` | **Main project**: PUG glycan-structure GNN pre-training and downstream tasks, based on the best checkpoint at epoch **94** | [README](./MASK/README.en.md) |
| `GT/` | **Comparison experiment**: extracts embeddings with GlycanGT for a fair comparison with PUG | [README](./GT/README.en.md) |
| `downstream/` | **Raw data and build scripts for downstream tasks** (source of each task's CSVs, see table below) | — |

> `MASK_i/` (the ablation study, same structure as MASK) is still running and will be uploaded once finished.

## Inside downstream/

| Subfolder | Description | README |
|-----------|-------------|--------|
| `tax(ML)/` | Taxonomy data (**clean version, no uncertain structures**, sourced from GlycanML): `allstata/` adds WURCS to the raw table, `3split.py` splits it into 8 levels under `split_files/`, used by `MASK/94/tax(ML)/` and the GT comparison | [README](./downstream/tax(ML)/README_EN.md) |
| `taxall/` | Taxonomy "all-taxa" data (contains **uncertain structures** that GlycanGT cannot process): `preprocessall.py` produces 8 level CSVs from superkingdom → species | [README](./downstream/taxall/README.en.md) |
| `imme/` | Immunogenicity data construction (data sourced from GlycanML, cleaned/merged into `imm_stata.csv`) | [README](./downstream/imme/README.md) |
| `linkpred/` | GlyConnect linking-type data construction (crawl/process, produces `glyconnect_structures.csv`) | [README](./downstream/linkpred/README.md) |
| `dic.csv` | The **original full** glycan dictionary downloaded from GlyTouCan; the `dic.csv` inside `MASK/` is the version after `deldownstata.py` removed leakage rows | — |
| `downstream_task_stats.py` / `.csv` / `.png` | Data-scale statistics (samples, labels) of downstream tasks and their plots | — |
| `report_second_round.py` / `second_round_report.md` | Analysis of second-round WURCS conversion failures (immunogenicity, taxonomy tasks) | — |

> The **initial CSVs of `tax(ML)/` and `imme/` both come from the GlycanML benchmark** (Xu et al., [*GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning*](https://arxiv.org/abs/2405.16206), ICLR 2025); `taxall/` data is crawled from Glycosmos by `preprocessall.py`.

## Naming note

The model was originally called **W0** and later renamed **PUG**; the code still uses the W0 naming (e.g., `glycanW0_classifier.py`, `get_embeddings_W0.py`). For the code-vs-paper naming of graph components (nodes/edges), see [MASK/README.en.md](./MASK/README.en.md).
