# MASK_i — Ablation: Removing the Pre-GAT "Edge → Node" Fusion Module

This folder is the ablation counterpart of the main experiment [MASK/](../MASK/README.md). It tests whether the module that aggregates probability-weighted edge features into node representations **before GAT** is actually useful. Except for the pre-trained weights, the workflow is identical to `MASK/`.

The ablation pre-training likewise happened to be best at **epoch 94**, so the downstream folder is named `94/` just like the main experiment — the two "94"s are an independent coincidence of separate training runs, and the **weights are not interchangeable**.

> 🌐 中文版: [README.md](./README.md)

## What was removed

Compared with the main experiment's `masktrain_windows_313.py`, only the following was deleted:

- `HeteroGNN._fuse_edge_features()` and its three projections `big_edge_fuse` / `small_edge_fuse` / `rev_small_edge_fuse`
- the call to it in `forward()` before GAT message passing

The removed module multiplied the edge embeddings of three edge types (`big_edge`, `small_edge`, `rev_small_edge`) by `edge_prob`, scattered them onto source/target nodes, and added them as a residual to the node representations. Without it, node representations are no longer directly enhanced by edge features before the first GAT layer.

Edge information is still passed through attention in `GATConv(edge_dim=...)`; `big_edge_prob_proj` / `small_edge_prob_proj` are unchanged. It can be verified by weight keys: 139 tensors in the ablation model vs. 145 in the main experiment.

## Layout

```
MASK_i/
├── masktrain_windows_313.py       # ablation pre-training (the only code differing from the main experiment)
├── wurcs_parser_reverse_edge.py   # same as the main experiment
├── output_all.json / output_all.txt / embedding_vocab.json   # reused parser outputs
├── checkpoints/best_model_mlm.pt  # weights trained with the ablated architecture (not interchangeable)
├── error_analysis/training_log.txt
└── 94/taxall/                     # downstream: taxonomy classification (counterpart of 94/taxall)
    ├── get_embeddings_W0.py
    ├── split_files/  split_files_processed_wurcs/
    ├── finaltrain/1..10/          # 10 random seeds
    ├── 总结.py
    └── analyze_graph_uncertainty.py
```

## Execution order (same as MASK)

```bash
# 1. Pre-training → checkpoints/best_model_mlm.pt
py -3.13 masktrain_windows_313.py

# 2. Copy the ablation weights into the downstream folder
copy checkpoints\best_model_mlm.pt 94\taxall\checkpoints\best_model_mlm.pt

# 3. Extract embeddings
cd 94\taxall
py -3.13 get_embeddings_W0.py \
    --input_dir ./split_files \
    --output_dir ./split_files_processed_wurcs \
    --vocab_path embedding_vocab.json \
    --checkpoint_path checkpoints/best_model_mlm.pt

# 4. Copy split_files_processed_wurcs/ into every seed folder under finaltrain

# 5. Batch training
cd finaltrain
py -3.13 "新建 文本文档.py"

# 6. Aggregate metrics → GlycanW0_metrics.csv
cd ..
py -3.13 总结.py

# 7. (Optional) uncertainty analysis → uncertainty_graph/
py -3.13 analyze_graph_uncertainty.py
```

## Notes

- Pre-training is complete: the best checkpoint was at **epoch 94**, and `checkpoints/best_model_mlm.pt` is that weight.
- Weights are not interchangeable: `get_embeddings_W0.py` loads with `strict=False`, so a mismatched checkpoint raises no error and silently ignores the missing fuse layers.
- See [94/taxall/README.md](./94/taxall/README.md) for downstream details.
- Among the large files, `dic.csv`, `checkpoints/*.pt` and `split_files/*.csv` are all **included in the repository** (below GitHub's 100 MB per-file limit); `output_all.json`/`output_all.txt` are pre-training products and are not uploaded — they are generated automatically on the first run.
