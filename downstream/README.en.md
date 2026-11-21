# downstream — Downstream Task Data & Build Scripts

> 🌐 中文版：[README.md](./README.md)

This folder holds the raw data, build scripts, and the glycan dictionary for the downstream tasks. See each subtask's own README for details:

| Subfolder | Description | README |
|-----------|-------------|--------|
| `tax(ML)/` | Taxonomy data (**clean version, no uncertain structures**, sourced from GlycanML); `3split.py` splits it by level into `split_files/` | [README](./tax(ML)/README_EN.md) |
| `taxall/` | Taxonomy "all-taxa" data (**sourced from Glycosmos**, **contains uncertain structures** that GlycanGT cannot process), 8 levels from superkingdom → species | [README](./taxall/README.en.md) |
| `imme/` | Immunogenicity data construction (data sourced from GlycanML), produces `imm_stata.csv` | [README](./imme/README.md) |
| `linkpred/` | GlyConnect linking-type data construction (crawl/process), produces `glyconnect_structures.csv` | [README](./linkpred/README.md) |

Top-level files:

- `dic.csv` — the **original full** glycan dictionary downloaded from GlyTouCan (leakage rows not removed); the `dic.csv` inside `MASK/` and `MASK_i/` is the version after running `deldownstata.py`
- `downstream_task_stats.csv / .png`, `plot_downstream_task_stats.py`, `report_second_round.py`, `second_round_report.md` — downstream task size statistics and report scripts
- `split_log.txt` — data splitting log
