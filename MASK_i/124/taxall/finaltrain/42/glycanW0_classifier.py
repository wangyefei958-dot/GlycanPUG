"""
GlycanGT 分类训练脚本 - 与 splittrain.py 流程一致

使用 GlycanGT 提取的 embedding 训练分类头，保持与 HeteroGNN 相同的训练配置：
- CB Focal Loss (Class-Balanced)
- AdamW optimizer
- MultiStageReduceLROnPlateau scheduler
- EMA-based early stopping
- Per-epoch threshold optimization

用于两个模型的公平比较。
"""

import os
import json
import random
import sys
import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss, confusion_matrix, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ========== Logger 类（与 splittrain.py 一致）==========

class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ========== 复制 splittrain.py 的关键组件 ==========

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        if isinstance(gamma, (int, float)):
            self.gamma = gamma
            self._gamma_vec = None
        else:
            self.gamma = None
            self._gamma_vec = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(gamma, dtype=torch.float)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        bce_loss = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)
        
        if self._gamma_vec is not None:
            gamma_t = self._gamma_vec.unsqueeze(0).expand_as(p_t)
            focal_factor = (1 - p_t) ** gamma_t
        else:
            focal_factor = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * 1.0
        else:
            alpha_t = 1.0

        loss = alpha_t * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_cb_weights(pos_counts, num_samples, beta=0.999):
    """
    计算Class-Balanced权重，用于 Focal Loss 的 alpha 参数。

    基于 CVPR 2019 "Class-Balanced Loss Based on Effective Number of Samples"。
    核心思想：用"有效样本数"替代原始样本数来衡量类别的信息量。

    effective_num = (1 - beta^n) / (1 - beta)
    raw_weight = (1 - beta) / (1 - beta^n) = 1 / effective_num

    CB 原始权重含义：样本多 → raw_weight 接近1，样本少 → raw_weight 接近0
    但 Focal Loss 的 alpha 需要：样本少 → alpha 大（放大正类信号）
    所以取倒数：alpha = 1 / raw_weight = effective_num
    再用 sqrt 平滑，避免极端值：alpha = sqrt(effective_num)

    最终效果：
    - 大样本类别：alpha ≈ 1（几乎不加权）
    - 小样本类别：alpha > 1（适度放大正类信号）
    - sqrt 保证不会过度放大

    Args:
        pos_counts: np.ndarray, 每个类别的正样本数
        num_samples: int, 总训练样本数（用于自动选beta）
        beta: float, 衰减系数

    Returns:
        torch.Tensor: per-class 正类权重，大样本≈1，小样本>1
    """
    import math
    weights = []
    for c in pos_counts:
        n = max(c, 1)  # 避免除零
        effective_num = (1.0 - beta ** n) / (1.0 - beta)
        # alpha = sqrt(effective_num)
        # 大样本: effective_num ≈ n, sqrt(n) 但我们归一化到最大类别=1
        # 小样本: effective_num 较小, 1/effective_num 较大
        w = math.sqrt(effective_num)
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float)
    # 归一化：最大类别(样本最多)的权重=1，其他类别按比例放大
    # 这样大样本类别 alpha=1，小样本类别 alpha>1
    max_w = weights.max()
    weights = weights / max_w
    # 取倒数：大样本→1，小样本→>1
    weights = 1.0 / weights
    return weights


class MultiStageReduceLROnPlateau:
    def __init__(self, optimizer, mode="max", milestones=(18, 36, 54),
                 factors=(0.5, 1/3, 1/4), min_lr=1e-6, threshold=1e-4):
        self.optimizer = optimizer
        self.mode = mode
        self.milestones = milestones
        self.factors = factors
        self.min_lr = min_lr
        self.threshold = threshold
        self.best_score = None
        self.num_bad_epochs = 0
        self.stage = 0
        self.last_reduced_at = 0

    def step(self, metrics):
        if self.best_score is None:
            self.best_score = metrics
            return

        if self.mode == "max":
            is_better = metrics > self.best_score + self.threshold
        else:
            is_better = metrics < self.best_score - self.threshold

        if is_better:
            self.best_score = metrics
            self.num_bad_epochs = 0
            self.stage = 0
        else:
            self.num_bad_epochs += 1

        if self.stage < len(self.milestones) and \
           self.num_bad_epochs >= self.milestones[self.stage]:
            self._reduce_lr(self.factors[self.stage])
            self.stage += 1

    def _reduce_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * factor, self.min_lr)
            param_group['lr'] = new_lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0, mode="min", ema_window=5, use_ema=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.ema_window = ema_window
        self.use_ema = use_ema
        self.best_score = None
        self.counter = 0
        self.ema_value = None
        self.ema_history = []

    def __call__(self, score):
        if self.use_ema:
            self.ema_history.append(score)
            if len(self.ema_history) > self.ema_window:
                self.ema_history.pop(0)
            self.ema_value = sum(self.ema_history) / len(self.ema_history)
            compare_score = self.ema_value
        else:
            compare_score = score

        if self.best_score is None:
            self.best_score = compare_score
            return False

        if self.mode == "min":
            is_better = compare_score < self.best_score - self.min_delta
        else:
            is_better = compare_score > self.best_score + self.min_delta

        if is_better:
            self.best_score = compare_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ========== GlycanGT 分类头模型 ==========

class GlycanGTClassifier(nn.Module):
    """
    基于 GlycanGT embedding 的分类头，结构参考 splittrain.py 的分类器设计
    """
    def __init__(self, input_dim=1027, hidden_dim=256, num_classes=None, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        

        # ✅ 固定三层结构（不再按类别数切换）
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)



# ========== 训练/评估函数 ==========

def train_one_epoch(model, dataloader, optimizer, criterion, monitor_criterion, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_monitor_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        monitor_loss = monitor_criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_monitor_loss += monitor_loss.item()
        num_batches += 1

    return total_loss / num_batches, total_monitor_loss / num_batches


def evaluate(model, dataloader, criterion, thresholds=None):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            num_batches += 1

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 使用 sigmoid 转换为概率（与 splittrain.py 一致）
    probs = 1 / (1 + np.exp(-all_logits))
    
    # 使用阈值进行预测
    if thresholds is not None:
        preds = apply_thresholds(probs, thresholds)
    else:
        # 默认使用 0.5 阈值（与 splittrain.py 一致：> 而不是 >=）
        preds = (probs > 0.5).astype(int)

    # 计算指标（与 splittrain.py 一致）
    accuracy = accuracy_score(all_labels, preds)
    f1_macro = f1_score(all_labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, preds, average='micro', zero_division=0)
    f1_samples = f1_score(all_labels, preds, average='samples', zero_division=0)
    precision_micro = precision_score(all_labels, preds, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, preds, average='micro', zero_division=0)
    precision_macro = precision_score(all_labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, preds, average='macro', zero_division=0)
    hl = hamming_loss(all_labels, preds)
    
    # 计算 macro AUC（与 splittrain.py 一致）
    num_classes = all_labels.shape[1]
    per_label_auc = []
    for i in range(num_classes):
        y_true = all_labels[:, i]
        y_score = probs[:, i]
        if len(np.unique(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_score))
                per_label_auc.append(auc)
            except ValueError:
                continue
    
    if per_label_auc:
        macro_auc = float(np.mean(per_label_auc))
    else:
        macro_auc = 0.0
    
    # 计算 micro AUC（将所有样本-标签对展平）
    try:
        micro_auc = float(roc_auc_score(all_labels.flatten(), probs.flatten()))
    except ValueError:
        micro_auc = 0.0

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_samples': f1_samples,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'hamming_loss': hl,
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
    }

    return total_loss / num_batches, metrics, preds, all_labels, probs


def find_optimal_thresholds(model, dataloader, num_classes, verbose=False):
    """
    基于训练集找到每个类别的最优阈值（与 splittrain.py 完全一致）。
    
    两阶段搜索：
    1. 粗搜索(步长0.05)在 [0.1, 0.9] 范围内定位大致范围
    2. 细搜索(步长0.01)在粗搜索最优值附近 ±0.1 范围内精确定位
    
    如果某类别全是正样本或全是负样本，使用默认阈值 0.5。
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)  # 与 splittrain.py 一致，直接用 sigmoid
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    optimal_thresholds = []

    for i in range(num_classes):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        # 检查并处理 NaN 值
        if np.isnan(y_score).any():
            print(f"Warning: NaN values found in class {i} probabilities, using default threshold 0.5")
            optimal_thresholds.append(0.5)
            continue

        # 如果该类别全是正样本或全是负样本，使用默认阈值
        if len(np.unique(y_true)) < 2:
            optimal_thresholds.append(0.5)
            continue

        # ---- 阶段1：粗搜索 (步长0.05) ----
        coarse_thresholds = np.arange(0.1, 0.91, 0.05)
        best_thresh_coarse = 0.5
        best_f1_coarse = 0.0
        for thresh in coarse_thresholds:
            y_pred = (y_score > thresh).astype(int)  # 与 splittrain.py 一致：> 而不是 >=
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1_coarse:
                best_f1_coarse = f1
                best_thresh_coarse = thresh

        # ---- 阶段2：细搜索 (步长0.01)，在粗搜索最优值附近 ----
        fine_start = max(0.1, best_thresh_coarse - 0.1)
        fine_end = min(0.9, best_thresh_coarse + 0.1)
        fine_thresholds = np.arange(fine_start, fine_end + 0.005, 0.01)

        best_thresh = best_thresh_coarse
        best_f1 = best_f1_coarse
        for thresh in fine_thresholds:
            y_pred = (y_score > thresh).astype(int)  # 与 splittrain.py 一致：> 而不是 >=
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds.append(best_thresh)

    return optimal_thresholds


def apply_thresholds(probs, thresholds):
    """使用 per-class 阈值进行预测（与 splittrain.py 一致）"""
    preds = np.zeros_like(probs)
    for i, thresh in enumerate(thresholds):
        preds[:, i] = (probs[:, i] > thresh).astype(int)
    return preds


def plot_confusion_matrices(all_labels, all_preds, label_classes, save_dir, fig_width_per_label=2.0, fig_height_per_label=2.0):
    """
    为多标签分类任务生成混淆矩阵图（与 splittrain.py 一致）。
    
    1. 生成一张大图，包含所有标签的 2x2 混淆矩阵，按标签出现次数降序排列
    2. 为每个标签单独生成一个 2x2 混淆矩阵图，保存到子文件夹中，用标签名命名
    
    Args:
        all_labels: np.ndarray, shape (N, num_classes), 真实标签（0/1）
        all_preds: np.ndarray, shape (N, num_classes), 预测标签（0/1）
        label_classes: list[str], 标签类别名称列表
        save_dir: str, 保存目录
        fig_width_per_label: float, 大图中每个子图的宽度（英寸）
        fig_height_per_label: float, 大图中每个子图的高度（英寸）
    """
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(save_dir, exist_ok=True)
    individual_dir = os.path.join(save_dir, "individual_labels")
    os.makedirs(individual_dir, exist_ok=True)

    num_classes = len(label_classes)

    # 计算每个标签的正样本数量，用于排序
    label_counts = []
    for i, cls_name in enumerate(label_classes):
        count = int(all_labels[:, i].sum())
        label_counts.append((i, cls_name, count))

    # 按标签出现次数降序排列
    label_counts.sort(key=lambda x: x[2], reverse=True)
    sorted_indices = [item[0] for item in label_counts]
    sorted_names = [item[1] for item in label_counts]
    sorted_counts = [item[2] for item in label_counts]

    # ---- 1. 生成大图：所有标签的混淆矩阵 ----
    # 计算网格布局
    n_cols = int(np.ceil(np.sqrt(num_classes)))
    n_rows = int(np.ceil(num_classes / n_cols))

    fig_width = n_cols * fig_width_per_label
    fig_height = n_rows * fig_height_per_label

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.suptitle("Confusion Matrices for All Labels (sorted by frequency)", 
                 fontsize=14, fontweight="bold", y=1.02)

    # 确保 axes 是二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for plot_idx in range(n_rows * n_cols):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]

        if plot_idx < num_classes:
            cls_idx = sorted_indices[plot_idx]
            cls_name = sorted_names[plot_idx]
            cls_count = sorted_counts[plot_idx]

            y_true = all_labels[:, cls_idx]
            y_pred = all_preds[:, cls_idx]

            # 2x2 混淆矩阵: [[TN, FP], [FN, TP]]
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"],
                        ax=ax, cbar=False, annot_kws={"size": 8})
            ax.set_title(f"{cls_name} (n={cls_count})", fontsize=7, pad=3)
            ax.tick_params(labelsize=6)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    big_fig_path = os.path.join(save_dir, "confusion_matrix_all_labels.png")
    fig.savefig(big_fig_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"All-labels confusion matrix saved to: {big_fig_path}")

    # ---- 2. 为每个标签单独生成 2x2 混淆矩阵图 ----
    for i, cls_name in enumerate(label_classes):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig_single, ax_single = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    ax=ax_single, cbar=True,
                    annot_kws={"size": 14})

        # 计算该标签的指标
        tn, fp, fn, tp = cm.ravel()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1)

        ax_single.set_title(f"Confusion Matrix - {cls_name}\n"
                          f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}",
                          fontsize=12)
        ax_single.tick_params(labelsize=10)

        # 清理文件名中的非法字符
        safe_name = cls_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        single_fig_path = os.path.join(individual_dir, f"confusion_matrix_{safe_name}.png")
        fig_single.savefig(single_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig_single)

    print(f"Individual label confusion matrices saved to: {individual_dir}/")


# ========== 数据加载 ==========

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_features_from_csv(csv_path: str, label_col: str = None):
    """
    从 CSV 文件加载 GlycanGT 特征。
    CSV 格式: wurcs, {层级名称如species/class/genus}, glycan, embedding (来自 get_embeddings.py 的输出)
    标签列优先使用传入的 label_col 参数（通常为文件名），否则自动识别
    返回: (embeddings, labels, unique_labels)
    """
    import csv
    import json
    
    embeddings = []
    labels_raw = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # 如果没有传入标签列名，则自动识别：排除已知的非标签列
        if label_col is None:
            known_columns = {'wurcs', 'glycan', 'embedding', 'glytoucan_id', 'id'}
            label_columns = [col for col in reader.fieldnames if col.lower() not in known_columns]
            
            if not label_columns:
                label_columns = [col for col in reader.fieldnames if col.lower() != 'wurcs' and col.lower() != 'glycan' and col.lower() != 'embedding']
            
            # 使用第一个找到的标签列
            label_col = label_columns[0] if label_columns else 'label'
        
        print(f"Using label column: '{label_col}'")
        
        for row in reader:
            # 读取 embedding 列
            if 'embedding' in row:
                emb = json.loads(row['embedding'])
                embeddings.append(emb)
            else:
                continue
            
            # 读取标签列
            if label_col in row:
                labels_raw.append(row[label_col])
            else:
                labels_raw.append('')
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # 获取所有唯一标签（按下划线拆分复合标签）
    all_single_labels = []
    for l in labels_raw:
        if l:
            if "_" in l:
                parts = l.split("_")
                all_single_labels.extend(parts)
            else:
                all_single_labels.append(l)
    
    unique_labels = sorted(set(all_single_labels))
    
    return embeddings, labels_raw, unique_labels


def load_hierarchy_data(data_dir, level):
    """加载指定层级的数据"""
    import csv
    filepath = os.path.join(data_dir, f"{level}.csv")
    glycan_list = []
    label_list = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            glycan_list.append(row['glycan'])
            label_list.append(row[level])
    
    return glycan_list, label_list


def extract_embeddings(glycan_list, output_dir='./outputs/glycangt_embeddings'):
    """使用 GlycanGT 提取 embedding"""
    import subprocess
    import tempfile
    
    # 创建临时输入文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('IUPAC Condensed,GlyTouCan ID\n')
        for i, glycan in enumerate(glycan_list):
            f.write(f'"{glycan}",glycan_{i:06d}\n')
        temp_input = f.name
    
    # 创建临时输出目录
    temp_output = tempfile.mkdtemp()
    
    try:
        # 运行 get_embeddings.py
        result = subprocess.run([
            'python', 'get_embeddings.py',
            '--input_dir', os.path.dirname(temp_input),
            '--output_dir', temp_output,
            '--iupac_col', 'IUPAC Condensed',
            '--id_col', 'GlyTouCan ID'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Embedding extraction failed: {result.stderr}")
            return None
        
        # 加载提取的 embedding
        embeddings = np.load(os.path.join(temp_output, 'embeddings.npy'))
        return embeddings
    finally:
        # 清理临时文件
        os.unlink(temp_input)
        import shutil
        shutil.rmtree(temp_output)


# ========== 主训练流程 ==========

def train_single_level(level, work_dir, batch_size=32, hidden_dim=128, dropout=0.3, lr=5e-4, weight_decay=1e-4):
    """
    训练单个层级的分类器
    """
    # 创建输出目录（直接放在工作目录下）
    ERROR_ANALYSIS_DIR = os.path.join(work_dir, f"training_results_{level}")
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)
    
    # 保存原始 stdout（与 splittrain.py 一致）
    original_stdout = sys.stdout
    
    # 重定向 stdout 到日志文件（与 splittrain.py 一致）
    log_path = os.path.join(ERROR_ANALYSIS_DIR, "training_log.txt")
    sys.stdout = Logger(log_path)

    print(f"\n{'='*60}")
    print(f"Training GlycanGT classifier for {level} level")
    print(f"{'='*60}")

    # 1. 从 work_dir 加载带有 embedding 的 CSV 文件
    print("Loading data...")
    csv_path = os.path.join(work_dir, f'{level}.csv')
    
    if os.path.exists(csv_path):
        print(f"Loading data from: {csv_path}")
        # 使用文件名（去掉扩展名）作为标签列名
        embeddings, label_list, unique_labels = load_features_from_csv(csv_path, label_col=level)
        
        num_classes = len(unique_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"Number of classes: {num_classes}")
        print(f"Embedding shape: {embeddings.shape}")
    else:
        print(f"[Error] CSV file not found: {csv_path}")
        print(f"Please run: py -3.13 get_embeddings.py first to generate embeddings")
        sys.stdout.log.close()
        return
    
    # 2. 准备标签（多标签格式，与 splittrain.py 一致）
    # 支持一个样本有多个标签，用下划线分隔，如 "Bacilli_Firmicutes"
    labels = np.zeros((len(label_list), num_classes), dtype=np.float32)
    for i, label_str in enumerate(label_list):
        if "_" in label_str:
            parts = label_str.split("_")
        else:
            parts = [label_str]
        for p in parts:
            if p in label_to_idx:
                labels[i, label_to_idx[p]] = 1.0


    # 3. 划分训练/验证/测试集（8:1:1，一步完成）
    print("Splitting data (80% train / 10% val / 10% test)...")

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


    # 一步直接切成 80% / 10% / 10%
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,      # 先切 20%
        random_state=SEED
    )

    train_idx, temp_idx = next(splitter.split(embeddings, labels))

    # 再把 20% 切成 10% val + 10% test
    splitter2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,     # 20% 的一半 = 10%
        random_state=SEED
    )

    val_idx, test_idx = next(splitter2.split(
        embeddings[temp_idx],
        labels[temp_idx]
    ))

    # ✅ 正确索引
    train_indices = train_idx.tolist()
    val_indices   = temp_idx[val_idx].tolist()
    test_indices  = temp_idx[test_idx].tolist()

    print(f"Train set: {len(train_indices)} samples")
    print(f"Val set: {len(val_indices)} samples")

    X_train, X_val, X_test = embeddings[train_indices], embeddings[val_indices], embeddings[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. 创建数据加载器
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    test_dataset = EmbeddingDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. 计算类别权重
    pos_counts = y_train.sum(axis=0)
    total_train = len(y_train)
    
    if total_train < 1000:
        CB_BETA = 0.9
    elif total_train < 10000:
        CB_BETA = 0.99
    elif total_train < 100000:
        CB_BETA = 0.999
    else:
        CB_BETA = 0.9999
    
    pos_weight = compute_cb_weights(pos_counts, total_train, beta=CB_BETA).to(DEVICE)
    FOCAL_GAMMA = 2.0

    # 输出类别权重信息（与 splittrain.py 一致）
    total_pos = pos_counts.sum()
    total_neg = num_classes * total_train - total_pos
    neg_weight_val = total_pos / max(total_neg, 1.0)
    
    print(f"\nClass weights (auto-computed from training set):")
    print(f"  Loss function: CB Focal Loss (Class-Balanced)")
    print(f"  CB beta: {CB_BETA} (auto-selected)")
    print(f"  Focal gamma: {FOCAL_GAMMA}")
    print(f"  neg_weight (global): {neg_weight_val:.4f}")
    for i, cls_name in enumerate(unique_labels):
        print(f"  {cls_name}: pos={int(pos_counts[i])}/{total_train}, "
              f"alpha={pos_weight[i]:.4f}")

    # 6. 创建模型
    model = GlycanGTClassifier(
        input_dim=1027,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    ).to(DEVICE)

    # 7. 损失函数和优化器
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=pos_weight, reduction="mean")
    monitor_criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=None, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 8. 学习率调度和早停
    scheduler = MultiStageReduceLROnPlateau(
        optimizer, mode="max", milestones=(18, 36, 54),
        factors=(0.5, 1/3, 1/4), min_lr=1e-6
    )
    early_stopper = EarlyStopping(patience=68, min_delta=1e-4, mode="max", 
                                   ema_window=5, use_ema=True)

    # 9. 训练循环
    MAX_EPOCHS = 1000
    best_val_f1 = 0.0
    best_model_state = None
    best_thresholds = None

    print(f"\nTraining config:")
    print(f"  Loss: CB Focal Loss (beta={CB_BETA}, gamma={FOCAL_GAMMA})")
    print(f"  Optimizer: AdamW, lr={lr}, weight_decay={weight_decay}")
    print(f"  Early stop patience: 68 (monitor: val_macro_auc EMA5)")
    print(f"  LR scheduler milestones: 18/36/54 (monitor: val_macro_auc)")
    print(f"  LR factors: 0.5 -> 1/3 -> 1/4")
    print(f"  Threshold search: on validation set (after training)")
    print("=" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_monitor_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, monitor_criterion)

        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, monitor_criterion)
        val_auc = val_metrics['macro_auc']

        stopped = early_stopper(val_auc)
        ema_auc = early_stopper.ema_value if early_stopper.ema_value is not None else val_auc
        scheduler.step(ema_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_auc > best_val_f1:
            best_val_f1 = val_auc
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())

        ema_str = f"{early_stopper.ema_value:.4f}" if early_stopper.ema_value is not None else "N/A"
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"Train Loss: {train_monitor_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Macro AUC: {val_auc:.4f} | "
              f"EMA5: {ema_str} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"ES: {early_stopper.counter}/68")

        if stopped:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print(f"Loading best model from epoch {best_epoch} (val_macro_auc={best_val_f1:.4f})")
    model.load_state_dict(best_model_state)

    # 在验证集上计算最优阈值（避免在训练集上搜索导致的过拟合）
    print("\nFinding optimal thresholds on validation set...")
    best_thresholds = find_optimal_thresholds(model, val_loader, num_classes)

    checkpoint_path = os.path.join(ERROR_ANALYSIS_DIR, "best_model.pt")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "val_macro_auc": best_val_f1,
        "label_classes": unique_labels,  # 已经是列表，不需要 tolist()
        "optimal_thresholds": best_thresholds,
    }, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
    test_loss, test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, monitor_criterion, thresholds=best_thresholds)
    
    # 同时用默认0.5阈值评估作为对比
    test_loss_05, test_metrics_05, _, _, _ = evaluate(
        model, test_loader, monitor_criterion, thresholds=None)

    print("\n" + "=" * 60)
    print(f"Test Results (with optimized thresholds from validation set):")
    print(f"  Thresholds used: {[f'{t:.2f}' for t in best_thresholds]}")
    print(f"  Loss:              {test_loss:.4f}")
    print(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"  ---")
    print(f"  F1 (micro):        {test_metrics['f1_micro']:.4f}")
    print(f"  F1 (macro):        {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (samples):      {test_metrics.get('f1_samples', 0.0):.4f}")
    print(f"  ---")
    print(f"  Precision (micro): {test_metrics.get('precision_micro', 0.0):.4f}")
    print(f"  Recall (micro):    {test_metrics.get('recall_micro', 0.0):.4f}")
    print(f"  Precision (macro): {test_metrics.get('precision_macro', 0.0):.4f}")
    print(f"  Recall (macro):    {test_metrics.get('recall_macro', 0.0):.4f}")
    print(f"  ---")
    print(f"  Hamming Loss:      {test_metrics['hamming_loss']:.4f}")
    print(f"  Macro AUC:         {test_metrics['macro_auc']:.4f}")
    print(f"  Micro AUC:         {test_metrics['micro_auc']:.4f}")
    
    # 计算总体混淆矩阵（与 splittrain.py 一致）
    tn_total = fp_total = fn_total = tp_total = 0
    for i in range(num_classes):
        cm = confusion_matrix(test_labels[:, i], test_preds[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tn_total += tn
        fp_total += fp
        fn_total += fn
        tp_total += tp

    print(f"\nOverall Confusion Matrix:")
    print(f"    TN: {tn_total:,}")
    print(f"    FP: {fp_total:,}")
    print(f"    FN: {fn_total:,}")
    print(f"    TP: {tp_total:,}")
    print(f"    Total: {tn_total + fp_total + fn_total + tp_total:,}")
    
    print(f"\n{'='*60}")
    print(f"Comparison: Optimized Thresholds vs Default 0.5 Threshold")
    print(f"{'='*60}")
    print(f"  Metric              | Optimized  | Default 0.5 | Improvement")
    print(f"  ---------------------|-----------|-------------|-------------")
    print(f"  F1 (macro)           | {test_metrics['f1_macro']:.4f}    | {test_metrics_05['f1_macro']:.4f}      | {test_metrics['f1_macro'] - test_metrics_05['f1_macro']:+.4f}")
    print(f"  F1 (micro)           | {test_metrics['f1_micro']:.4f}    | {test_metrics_05['f1_micro']:.4f}      | {test_metrics['f1_micro'] - test_metrics_05['f1_micro']:+.4f}")
    print(f"  Accuracy             | {test_metrics['accuracy']:.4f}    | {test_metrics_05['accuracy']:.4f}      | {test_metrics['accuracy'] - test_metrics_05['accuracy']:+.4f}")
    print(f"  Precision (macro)    | {test_metrics['precision_macro']:.4f}    | {test_metrics_05['precision_macro']:.4f}      | {test_metrics['precision_macro'] - test_metrics_05['precision_macro']:+.4f}")
    print(f"  Recall (macro)       | {test_metrics['recall_macro']:.4f}    | {test_metrics_05['recall_macro']:.4f}      | {test_metrics['recall_macro'] - test_metrics_05['recall_macro']:+.4f}")
    print("=" * 60)

    # 12. 生成混淆矩阵图（与 splittrain.py 一致）
    print("\n" + "=" * 60)
    print("Generating confusion matrices...")
    confusion_matrix_dir = os.path.join(ERROR_ANALYSIS_DIR, "confusion_matrices")
    plot_confusion_matrices(
        all_labels=test_labels,
        all_preds=test_preds,
        label_classes=unique_labels,
        save_dir=confusion_matrix_dir,
        fig_width_per_label=2.0,
        fig_height_per_label=2.0
    )

    # 13. 保存模型和阈值
    print("\n" + "=" * 60)
    print("Saving model and thresholds...")
    torch.save(best_model_state, os.path.join(ERROR_ANALYSIS_DIR, "best_model.pth"))
    np.save(os.path.join(ERROR_ANALYSIS_DIR, "best_thresholds.npy"), best_thresholds)

    print(f"\nTraining complete for {level} level!")
    print(f"Results saved to: {ERROR_ANALYSIS_DIR}")
    
    # 关闭日志文件并恢复原始 stdout（与 splittrain.py 一致）
    sys.stdout.log.close()
    sys.stdout = original_stdout


def main():
    import argparse
    import glob
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default=os.path.join(script_dir, 'split_files_processed_wurcs'),
                        help='Working directory containing CSV files with embeddings (output from get_embeddings.py)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    # 自动查找 work_dir 下的所有 CSV 文件
    csv_pattern = os.path.join(args.work_dir, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"[Error] No CSV files found in {args.work_dir}")
        print(f"Please run: py -3.13 get_embeddings.py first to generate embeddings")
        return
    
    # 提取层级名称（去掉 .csv 后缀）
    levels = []
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        level = os.path.splitext(basename)[0]
        # 排除可能的临时文件或非层级文件
        if level not in ['__pycache__', '.DS_Store', 'Thumbs']:
            levels.append(level)
    
    if not levels:
        print(f"[Error] No valid level CSV files found in {args.work_dir}")
        return
    
    print(f"Found {len(levels)} levels to train: {', '.join(levels)}")
    
    # 逐个训练每个层级
    for level in levels:
        print(f"\n{'='*80}")
        print(f"Starting training for {level} level...")
        print(f"{'='*80}")
        train_single_level(
            level=level,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"\nTraining for {level} level completed!\n")
    
    print(f"\n{'='*80}")
    print("All training completed!")
    print(f"Results saved in {args.work_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
