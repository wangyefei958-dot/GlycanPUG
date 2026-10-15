# MASK补充 (MASK_i): Ablation Study

> **Naming note**: The model was originally called **W0** and later renamed **PUG**, but the code still uses the W0 naming (e.g., `glycanW0_classifier.py`, `get_embeddings_W0.py`).

> **Code vs. paper naming**: The code retains the original naming, which differs slightly from the paper — `big_nodes` are called "glycosyl residue nodes" in the paper, `small_nodes` → "substituent nodes", `big_edges` → "inter-residue glycosidic edges", `small_edges` → "substituent attachment edges", and the `stereo_type` feature → "carbon descriptor". Other names are essentially unchanged.

This folder is the **ablation study version** of the main project `MASK/`. Its directory structure and scripts are identical to `MASK/`, and **the usage is the same** — see the [main MASK README](../MASK/README.en.md).

> 🌐 中文版：[README.md](./README.md)

## Difference from MASK

During pre-training, the **checkpoint at epoch 124 was the best model** (the main MASK project uses epoch 95), so the downstream task folder is named `124`:

```
MASK补充/
├── wurcs_parser_reverse_edge.py   # WURCS parser (same as MASK)
├── masktrain_windows_313.py       # MLM pre-training (same as MASK)
├── deldownstata.py                # Data-leakage cleanup tool (same as MASK)
├── dic.csv                         # Main data dictionary
├── checkpoints/
│   └── best_model_mlm.pt          # 124-epoch pre-training weights
├── downstreamstata/                # Downstream statistics / leakage-check CSVs
└── 124/                            # Downstream tasks based on the "best checkpoint at 124 epochs"
    └── taxall/                     # Taxonomy "all-taxa" configuration (incl. superkingdom, 9 levels)
        ├── get_embeddings_W0.py
        ├── glycanW0_classifier.py
        ├── 总结.py
        ├── embedding_vocab.json
        ├── GlycanW0_metrics.csv
        ├── split_files/             # WURCS CSVs split by taxonomic level
        └── finaltrain/              # Multi-seed final training (42 / 58 / 518 / 5188 / 51888)
```

> **Note**: The `dic.csv` in this folder has already been processed by `deldownstata.py` (leakage rows removed); the original version downloaded from GlyTouCan is in the `downstream/` folder.

## Execution

Pre-training and downstream tasks run exactly the same way as in `MASK/` — see the [main MASK README's "Execution Order"](../MASK/README.en.md#execution-order-main-folder) and ["About the 95 folder"](../MASK/README.en.md#about-the-95-folder) sections. The only difference is that downstream tasks are carried out under `124/`, and this ablation study currently only ran the `taxall` task.
