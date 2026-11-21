# taxall — 消融实验的下游：物种分类（8 层级，种子 1–10）

本目录是 [MASK_i/README.md](../README.md) 所述消融实验的下游部分，对应主实验 `MASK/94/taxall`（消融预训练同样恰好在第 94 轮最优，故同样以 `94` 命名）。**预训练必须使用消融版 `masktrain_windows_313.py` 训练出的权重**（见上一级 `checkpoints/best_model_mlm.pt`），不能用主实验权重。

## 目录结构

```
94/taxall/
├── get_embeddings_W0.py            # 提取 1027 维嵌入
├── 总结.py                          # 汇总多 seed 的 F1/AUC → GlycanW0_metrics.csv
├── analyze_graph_uncertainty.py     # 图结构不确定性分析（在本目录运行）
├── embedding_vocab.json
├── checkpoints/best_model_mlm.pt   # 需放消融权重
├── split_files/                     # 8 个层级：superkingdom / kingdom / phylum / class / order / family / genus / species
├── split_files_processed_wurcs/
├── uncertainty_graph/
└── finaltrain/1..10/                # 10 个种子，各自含 glycanW0_classifier.py
```

## 运行顺序

```bash
# 1. 提取嵌入
py -3.13 get_embeddings_W0.py \
    --input_dir ./split_files \
    --output_dir ./split_files_processed_wurcs \
    --vocab_path embedding_vocab.json \
    --checkpoint_path checkpoints/best_model_mlm.pt

# 2. 将 split_files_processed_wurcs/ 复制到 finaltrain 下每个种子文件夹

# 3. 批量训练
cd finaltrain
py -3.13 "新建 文本文档.py"

# 4. 汇总 → GlycanW0_metrics.csv
cd ..
py -3.13 总结.py

# 5. （可选）不确定性分析 → uncertainty_graph/
py -3.13 analyze_graph_uncertainty.py
```

`总结.py` 与 `analyze_graph_uncertainty.py` 都会自动识别 `finaltrain/` 下以数字命名的种子目录。

## 注意

- `checkpoints/best_model_mlm.pt` 必须是消融结构（139 个权重张量）训练出的权重；`get_embeddings_W0.py` 用 `strict=False` 加载，权重不对不会报错。
- 大文件（`split_files/*.csv`、`checkpoints/*.pt`）均**已随仓库上传**，克隆后可直接运行；`split_files_processed_wurcs/*.csv` 为嵌入提取步骤的生成产物，未上传，首次运行会自动生成，生成后按运行顺序第 2 步手动复制到 `finaltrain/` 下各种子文件夹；其中的 `training_results_*/` 训练结果同样未上传，由第 3 步分类训练生成。
