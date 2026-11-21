# MASK_i — 消融实验：移除 GAT 前的「边 → 节点」融合模块

本目录是主实验 [MASK/](../MASK/README.md) 的消融对照，用于验证「在 GAT 之前把概率加权后的边特征聚合进节点表示」这一模块的作用。除预训练权重外，运行方法与 `MASK/` 完全一致。

消融预训练同样恰好在第 **94 轮** checkpoint 取得最优，因此下游文件夹与主实验一样命名为 `94/`——两个「94」是各自独立训练的巧合，**权重互不通用**。

> 🌐 English version: [README.en.md](./README.en.md)

## 改动内容

相对主实验的 `masktrain_windows_313.py`，只删除了：

- `HeteroGNN._fuse_edge_features()` 及其 3 个投影层 `big_edge_fuse` / `small_edge_fuse` / `rev_small_edge_fuse`
- `forward()` 中在 GAT 消息传递之前对它的调用

该模块原本把三类边（`big_edge`、`small_edge`、`rev_small_edge`）的边嵌入乘以 `edge_prob` 后按源/目标节点累加，残差加到节点表示上。删除后，节点表示进入第一层 GAT 前不再被边特征直接增强。

边信息仍通过 `GATConv(edge_dim=...)` 的注意力参与传递，`big_edge_prob_proj` / `small_edge_prob_proj` 保留不变。权重键数可核对：消融模型 139 个张量，主实验 145 个。

## 目录结构

```
MASK_i/
├── masktrain_windows_313.py       # 消融版预训练（唯一与主实验不同的代码）
├── wurcs_parser_reverse_edge.py   # 与主实验相同
├── output_all.json / output_all.txt / embedding_vocab.json   # 沿用主实验解析产物
├── checkpoints/best_model_mlm.pt  # 消融结构训练权重（不可与主实验权重混用）
├── error_analysis/training_log.txt
└── 94/taxall/                     # 下游：物种分类（对应主实验 94/taxall）
    ├── get_embeddings_W0.py
    ├── split_files/  split_files_processed_wurcs/
    ├── finaltrain/1..10/          # 10 个随机种子
    ├── 总结.py
    └── analyze_graph_uncertainty.py
```

## 运行顺序（与 MASK 相同）

```bash
# 1. 预训练 → checkpoints/best_model_mlm.pt
py -3.13 masktrain_windows_313.py

# 2. 把消融权重放到下游目录
copy checkpoints\best_model_mlm.pt 94\taxall\checkpoints\best_model_mlm.pt

# 3. 提取嵌入
cd 94\taxall
py -3.13 get_embeddings_W0.py \
    --input_dir ./split_files \
    --output_dir ./split_files_processed_wurcs \
    --vocab_path embedding_vocab.json \
    --checkpoint_path checkpoints/best_model_mlm.pt

# 4. 将 split_files_processed_wurcs/ 复制到 finaltrain 下每个种子文件夹

# 5. 批量训练
cd finaltrain
py -3.13 "新建 文本文档.py"

# 6. 汇总指标 → GlycanW0_metrics.csv
cd ..
py -3.13 总结.py

# 7. （可选）不确定性分析 → uncertainty_graph/
py -3.13 analyze_graph_uncertainty.py
```

## 注意

- 预训练已完成：第 **94 轮** checkpoint 最优，`checkpoints/best_model_mlm.pt` 即为该轮权重。
- 权重不可互换：`get_embeddings_W0.py` 用 `strict=False` 加载，键不匹配时不会报错，只会静默忽略缺失的 fuse 层。
- 下游细节见 [94/taxall/README.md](./94/taxall/README.md)。
- 大文件中 `dic.csv`、`checkpoints/*.pt`、`split_files/*.csv` 均**已随仓库上传**（低于 GitHub 100 MB 单文件限制）；`output_all.json`/`output_all.txt` 为预训练生成产物，未上传，首次运行会自动生成。
