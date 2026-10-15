# -*- coding: utf-8 -*-
import json
import os
import random
from collections import defaultdict
from copy import deepcopy
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, global_mean_pool, global_max_pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, confusion_matrix, roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 混合精度训练支持
from torch.amp import autocast, GradScaler

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ALL_PATH = os.path.join(DATA_DIR, "output_all.json")
VOCAB_PATH = os.path.join(DATA_DIR, "embedding_vocab.json")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
ERROR_ANALYSIS_DIR = os.path.join(DATA_DIR, "error_analysis")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)


def compute_task_class_weights(train_subset, token_to_idx, vocab_sizes, num_samples_for_stats=None):
    """
    统计训练集中每个特征的类别分布，计算类别平衡权重
    
    Args:
        train_subset: 训练集Subset对象
        token_to_idx: 特征名到类别映射的字典
        vocab_sizes: 各特征的词汇表大小字典
        num_samples_for_stats: 用于统计的样本数量（None表示全部）
    
    Returns:
        dict: 每个特征的per-class权重，形状为 (num_classes,)
    """
    import collections
    
    # 要统计的特征及其对应的数据源
    feat_to_source = {
        # "anomeric_position": ("big_node", 1),  # 已删除，不再作为预测任务（成环方式会泄露位置信息）
        "anomeric_config": ("big_node", 2),
        "substituent": ("small_node", 0),  # 取代基预测任务，O不参与loss计算
        "big_edge": ("big_edge_type", None),  # 边特征
        "small_edge_stereo": ("stereo_type", None),  # 边特征
        "ring_bond": ("ring_bond_type", None),  # ring_bond多标签预测，只对概率为1的计算损失
    }
    
    task_class_weights = {}
    
    # 如果设置了采样数量，随机采样
    indices = list(range(len(train_subset)))
    if num_samples_for_stats is not None and num_samples_for_stats < len(indices):
        import random
        random.seed(42)
        indices = random.sample(indices, num_samples_for_stats)
    
    print("\n  Computing class distribution for Focal Loss...")
    
    # 特征名称映射到 vocab_sizes 的键名（提前定义，后面会用到）
    feat_name_to_vocab_key = {
        # "anomeric_position": "anomeric_position_type",  # 已删除，不再作为预测任务
        "anomeric_config": "anomeric_config_type",
        "substituent": "substituent",
        "big_edge": "big_edge_type",
        "small_edge_stereo": "stereo_type",
        "ring_bond": "ring_bond_type",  # ring_bond多标签预测
    }
    
    # 初始化计数器
    class_counts = {k: collections.defaultdict(int) for k in feat_to_source.keys()}
    
    # 获取每个特征的MASK token索引和不可预测标签索引
    mask_indices = {}
    unpredictable_indices_dict = {}
    for feat_name in feat_to_source.keys():
        # 获取vocab_key
        vocab_key = feat_name_to_vocab_key.get(feat_name, feat_name)
        # MASK token索引
        mask_idx = get_mask_token_idx(token_to_idx, vocab_key) if token_to_idx else -1
        mask_indices[feat_name] = mask_idx
        # 不可预测标签索引
        unpredictable_indices_dict[feat_name] = get_unpredictable_label_indices(token_to_idx, feat_name) if token_to_idx else set()
    
    # 遍历训练集统计类别分布
    for idx in indices:
        graph, _, _, _ = train_subset.dataset[train_subset.indices[idx]]
        
        # 节点特征统计
        big_x = graph["big_node"].x
        small_x = graph["small_node"].x
        
        for feat_name, (node_type, col) in feat_to_source.items():
            mask_idx = mask_indices.get(feat_name, -1)
            unpredictable_indices = unpredictable_indices_dict.get(feat_name, set())
            
            if node_type == "big_node" and col is not None:
                # 统计节点特征的类别分布（统计所有值，排除MASK和不可预测标签）
                values = big_x[:, col].numpy()
                for v in values:
                    v_int = int(v)
                    # 排除MASK token和不可预测标签
                    if v_int != mask_idx and v_int not in unpredictable_indices:
                        class_counts[feat_name][v_int] += 1
            elif node_type == "small_node":
                values = small_x[:, col].numpy()
                for v in values:
                    v_int = int(v)
                    # 排除MASK token和不可预测标签
                    if v_int != mask_idx and v_int not in unpredictable_indices:
                        class_counts[feat_name][v_int] += 1
            elif node_type == "big_edge_type":
                # big_edge_type 在 edge_attr 中
                if ("big_node", "big_edge", "big_node") in graph.edge_types:
                    edge_attr = graph[("big_node", "big_edge", "big_node")].edge_attr
                    if edge_attr is not None and edge_attr.size(1) > 0:
                        for v in edge_attr[:, 0].numpy():
                            v_int = int(v)
                            # 排除MASK token和不可预测标签
                            if v_int != mask_idx and v_int not in unpredictable_indices:
                                class_counts[feat_name][v_int] += 1
            elif node_type == "stereo_type":
                # stereo_type 在 edge_attr 中
                if ("big_node", "small_edge", "small_node") in graph.edge_types:
                    edge_attr = graph[("big_node", "small_edge", "small_node")].edge_attr
                    if edge_attr is not None and edge_attr.size(1) > 0:
                        for v in edge_attr[:, 0].numpy():
                            v_int = int(v)
                            # 排除MASK token和不可预测标签
                            if v_int != mask_idx and v_int not in unpredictable_indices:
                                class_counts[feat_name][v_int] += 1
            elif node_type == "ring_bond_type":
                # ring_bond_type 是多标签任务，存储在图的 rb_type_indices 中
                if hasattr(graph, 'rb_type_indices') and hasattr(graph, 'rb_probs'):
                    rb_types = graph.rb_type_indices.numpy()
                    rb_probs = graph.rb_probs.numpy()
                    # 只统计概率为1且不带问号的环键（这些才参与训练）
                    valid_mask = (rb_probs >= 1.0)
                    if hasattr(graph, 'rb_has_question'):
                        rb_has_q = graph.rb_has_question.numpy()
                        valid_mask = valid_mask & (~rb_has_q)
                    valid_types = rb_types[valid_mask]
                    for v in valid_types:
                        v_int = int(v)
                        # 排除MASK token和不可预测标签
                        if v_int != mask_idx and v_int not in unpredictable_indices:
                            class_counts[feat_name][v_int] += 1
    
    
    for feat_name, counts in class_counts.items():
        vocab_key = feat_name_to_vocab_key.get(feat_name, feat_name)
        num_classes = vocab_sizes.get(vocab_key, 10)
        # 使用 compute_cb_weights 计算平衡权重
        # 注意：只计算实际存在的类别的权重，不包含MASK token和不可预测标签
        pos_counts = np.array([counts.get(i, 0) for i in range(num_classes)])
        # 对于样本数为0的类别（包括MASK token和不可预测标签），权重设为0（不参与损失计算）
        # 但为了避免除零，至少设为1
        pos_counts = np.maximum(pos_counts, 1)
        
        weights = compute_cb_weights(pos_counts, pos_counts.sum(), beta=0.999)
        # 对于MASK token和不可预测标签，权重设为0
        mask_idx = mask_indices.get(feat_name, -1)
        unpredictable_indices = unpredictable_indices_dict.get(feat_name, set())
        if mask_idx >= 0 and mask_idx < len(weights):
            weights[mask_idx] = 0.0
        for idx in unpredictable_indices:
            if idx < len(weights):
                weights[idx] = 0.0
        
        task_class_weights[feat_name] = weights.to(DEVICE)
        
        # 打印类别分布
        total = pos_counts.sum()
        print(f"    {feat_name}: {num_classes} classes, total={total}")
        for i, (c, w) in enumerate(zip(pos_counts, weights.cpu().numpy())):
            if c > 0:
                print(f"      Class {i}: {c} ({c/total*100:.1f}%), weight={w:.2f}")
    
    return task_class_weights


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab_sizes = {}
    token_to_idx = {}
    for feat_name, mapping in raw.items():
        vocab_sizes[feat_name] = len(mapping)
        token_to_idx[feat_name] = mapping
    return vocab_sizes, token_to_idx


def get_mask_token_idx(token_to_idx, vocab_key):
    """
    获取指定特征的MASK token索引。
    
    Args:
        token_to_idx: dict, 特征名到类别映射的字典
        vocab_key: str, vocab中的特征名称（如 "carbon_count_type", "substituent"等）
    
    Returns:
        int: MASK token的索引，如果不存在则返回vocab_size（最后一个索引）
    """
    if vocab_key in token_to_idx:
        vocab_mapping = token_to_idx[vocab_key]
        # MASK token名称为 "<MASK>"
        if "<MASK>" in vocab_mapping:
            return vocab_mapping["<MASK>"]
        # 如果没有MASK token，返回vocab_size作为MASK索引
        return len(vocab_mapping)
    return -1  # 不存在的特征


def load_all_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_label_classes(raw_data):
    """从数据中自动提取所有唯一的标签类别"""
    all_labels = set()
    for sample in raw_data:
        label_str = sample["label"]
        if "_" in label_str:
            parts = label_str.split("_")
        else:
            parts = [label_str]
        for p in parts:
            all_labels.add(p)
    return sorted(all_labels)


# 默认为空列表，将在 main() 中从数据自动提取并覆盖
LABEL_CLASSES = []


def get_unpredictable_label_indices(token_to_idx, task_name):
    """
    根据vocab中的标签名称，获取不可预测的标签索引列表。
    
    不可预测的标签规则：
    - 包含 '?' 的标签（表示未知位置）
    - 'x' 标签（表示未知构型）
    - 'unknown' 标签（表示未知碳数）
    
    Args:
        token_to_idx: dict, 特征名到类别映射的字典
        task_name: str, 任务名称（如 "carbon_count", "anomeric_config", "big_edge", "small_edge_stereo"）
    
    Returns:
        set: 不可预测的标签索引集合
    """
    # 任务名称映射到 vocab 中的特征名称
    task_to_vocab_key = {
        "carbon_count": "carbon_count_type",
        "anomeric_config": "anomeric_config_type",
        "substituent": "substituent",     # O为无需预测项
        "big_edge": "big_edge_type",
        "small_edge_stereo": "stereo_type",
        "big_edge_exists": None,  # 边存在性预测是二分类，没有不可预测标签
        "ring_bond": "ring_bond_type",    # ring_bond多标签预测
    }
    
    vocab_key = task_to_vocab_key.get(task_name)
    if vocab_key is None or vocab_key not in token_to_idx:
        return set()
    
    unpredictable_indices = set()
    vocab_mapping = token_to_idx[vocab_key]
    
    for label_name, label_idx in vocab_mapping.items():
        # 判断是否为不可预测标签
        is_unpredictable = False
        
        # 包含 '?' 的标签
        if "?" in label_name:
            is_unpredictable = True
        
        # 'x' 标签（小写x表示未知构型）
        if label_name == "x":
            is_unpredictable = True
        
        # 'unknown' 标签
        if label_name == "unknown":
            is_unpredictable = True
        
        # 'N/A' 标签（表示没有成环或没有相关信息）
        if label_name == "N/A":
            is_unpredictable = True
        
        if is_unpredictable:
            unpredictable_indices.add(label_idx)
    
    # substituent 特殊处理："O"为无需预测项
    if task_name == "substituent":
        # 在 vocab 中查找 "O" 对应的索引
        if "O" in vocab_mapping:
            unpredictable_indices.add(vocab_mapping["O"])
    
    # 所有任务：<MASK> token为无需计算项
    if "<MASK>" in vocab_mapping:
        unpredictable_indices.add(vocab_mapping["<MASK>"])
    
    return unpredictable_indices


def compute_task_metrics(all_preds, all_targets, task_names, vocab_sizes=None, token_to_idx=None):
    """
    计算每个任务的分类指标（准确率、macro F1、召回率、精确率、每个标签的F1）
    
    改进版本：支持不可预测标签的标记，索引按照vocab的来，支持多标签任务。

    Args:
        all_preds: dict, 每个任务的预测张量列表
        all_targets: dict, 每个任务的真实标签张量列表
        task_names: list, 任务名称列表
        vocab_sizes: dict, 每个任务的词汇表大小（可选，用于确保所有标签都被计算）
        token_to_idx: dict, 特征名到类别映射的字典（可选，用于获取标签名称和判断不可预测标签）

    Returns:
        dict: 每个任务的指标字典，格式如 
        {"task_name": {
            "accuracy": x, 
            "macro_f1": y, 
            "recall": z, 
            "precision": w, 
            "per_class_f1": {0: f1_0, 1: f1_1, ...}, 
            "per_class_count": {0: count_0, 1: count_1, ...},
            "unpredictable_labels": {idx: label_name, ...},
            "label_names": {idx: label_name, ...}
        }}
    """
    metrics = {}
    
    # 任务名称映射到 vocab 中的特征名称
    task_to_vocab_key = {
        "carbon_count": "carbon_count_type",
        "anomeric_config": "anomeric_config_type",
        "substituent": "substituent",
        "big_edge": "big_edge_type",
        "small_edge_stereo": "stereo_type",
        "big_edge_exists": None,
        "ring_bond": "ring_bond_type",    # ring_bond多标签预测
        "is_cyclic": None,                # is_cyclic是二分类任务，不需要vocab
    }
    
    # 多标签任务列表
    multi_label_tasks = {"ring_bond"}
    
    # 二分类任务列表（不需要计算per_class_f1，只需要整体指标）
    binary_tasks = {"big_edge_exists", "is_cyclic"}
    
    for task_name in task_names:
        if task_name not in all_preds or task_name not in all_targets:
            continue
        preds = torch.cat(all_preds[task_name], dim=0) if all_preds[task_name] else torch.tensor([])
        targets = torch.cat(all_targets[task_name], dim=0) if all_targets[task_name] else torch.tensor([])

        if len(preds) == 0 or len(targets) == 0:
            metrics[task_name] = {
                "accuracy": 0.0, 
                "macro_f1": 0.0, 
                "recall": 0.0, 
                "precision": 0.0, 
                "count": 0, 
                "per_class_f1": {}, 
                "per_class_count": {},
                "unpredictable_labels": {},
                "label_names": {},
                "hamming_loss": 0.0
            }
            continue

        # 转换为 numpy
        preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

        # 获取 vocab 信息
        vocab_key = task_to_vocab_key.get(task_name)
        
        # 特殊任务：二分类任务（big_edge_exists 和 is_cyclic），不是从 vocab 获取类别数
        if task_name in binary_tasks:
            # 从实际数据中获取类别数
            all_labels = np.concatenate([preds_np, targets_np]) if len(preds_np) > 0 else np.array([])
            if len(all_labels) > 0:
                num_classes = int(all_labels.max()) + 1
            else:
                num_classes = 2
            if task_name == "big_edge_exists":
                label_names = {0: "not_exists", 1: "exists"}
            elif task_name == "is_cyclic":
                label_names = {0: "not_cyclic", 1: "cyclic"}
            else:
                label_names = {0: "negative", 1: "positive"}
            unpredictable_labels = {}
        else:
            num_classes = vocab_sizes.get(vocab_key, vocab_sizes.get(task_name, 10)) if vocab_sizes else 10
        
        # 获取不可预测的标签索引
        unpredictable_indices = get_unpredictable_label_indices(token_to_idx, task_name) if token_to_idx else set()
        
        # 获取标签名称映射（对于 big_edge_exists 和 is_cyclic 已经设置了，跳过）
        if task_name != "big_edge_exists" and task_name != "is_cyclic":
            label_names = {}
            unpredictable_labels = {}
            if token_to_idx and vocab_key and vocab_key in token_to_idx:
                vocab_mapping = token_to_idx[vocab_key]
                # 反转映射：索引 -> 名称
                idx_to_name = {v: k for k, v in vocab_mapping.items()}
                for idx in range(num_classes):
                    label_names[idx] = idx_to_name.get(idx, f"Label_{idx}")
                    if idx in unpredictable_indices:
                        unpredictable_labels[idx] = label_names[idx]

        # 多标签任务处理
        if task_name in multi_label_tasks:
            # 多标签任务：计算 hamming loss, macro f1, micro f1
            
            # 获取不可预测的标签索引
            unpredictable_indices = get_unpredictable_label_indices(token_to_idx, task_name) if token_to_idx else set()
            
            # 更新 unpredictable_labels 以包含最新的 unpredictable_indices
            for idx in unpredictable_indices:
                if idx in label_names:
                    unpredictable_labels[idx] = label_names[idx]
            
            # 过滤掉不可预测的标签（只保留可预测的标签列）
            valid_cols = [idx for idx in range(num_classes) if idx not in unpredictable_indices]
            if valid_cols:
                filtered_targets = targets_np[:, valid_cols]
                filtered_preds = preds_np[:, valid_cols]
            else:
                filtered_targets = targets_np
                filtered_preds = preds_np
            
            # 计算指标（使用过滤后的数据）
            hamming = hamming_loss(filtered_targets, filtered_preds)
            macro_f1 = f1_score(filtered_targets, filtered_preds, average='macro', zero_division=0)
            micro_f1 = f1_score(filtered_targets, filtered_preds, average='micro', zero_division=0)
            recall = recall_score(filtered_targets, filtered_preds, average='macro', zero_division=0)
            precision = precision_score(filtered_targets, filtered_preds, average='macro', zero_division=0)
            
            # 计算每个类别的F1（包括所有类别，但标记不可预测的为None）
            per_class_f1_values = f1_score(targets_np, preds_np, average=None, zero_division=0)
            per_class_f1_dict = {}
            for idx, f1 in enumerate(per_class_f1_values):
                if idx in unpredictable_indices:
                    per_class_f1_dict[idx] = None  # 不可预测的标签标记为无需计算
                else:
                    per_class_f1_dict[idx] = float(f1)
            
            # 计算每个类别的样本数（正样本数）
            per_class_count = {}
            for idx in range(num_classes):
                count = np.sum(targets_np[:, idx] == 1)
                per_class_count[idx] = int(count)
            
            # 计算总的标签数（只统计可预测标签的正样本数之和，排除不可预测标签）
            total_label_count = sum(count for idx, count in per_class_count.items() 
                                   if idx not in unpredictable_indices)
            
            metrics[task_name] = {
                "accuracy": micro_f1,  # 用 micro F1 作为多标签任务的准确率
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "recall": recall,
                "precision": precision,
                "count": total_label_count,  # 总的标签数（正样本数之和）
                "per_class_f1": per_class_f1_dict,
                "per_class_count": per_class_count,
                "unpredictable_labels": unpredictable_labels,
                "label_names": label_names,
                "hamming_loss": hamming
            }
            continue
        
        # 单标签任务处理
        # 计算准确率
        accuracy = accuracy_score(targets_np, preds_np)

        # 计算每个标签的样本数量（按vocab索引）
        per_class_count = {}
        for idx in range(num_classes):
            count = np.sum(targets_np == idx)
            per_class_count[idx] = int(count)

        # 计算每个标签的 F1 分数（按vocab索引）
        # 使用 labels 参数确保所有标签都被计算，索引按vocab顺序
        labels = list(range(num_classes))
        per_class_f1_values = f1_score(targets_np, preds_np, labels=labels, average=None, zero_division=0)
        
        # 构建 per_class_f1 字典
        per_class_f1_dict = {}
        for idx, f1 in enumerate(per_class_f1_values):
            if idx in unpredictable_indices:
                # 不可预测的标签，标记为 "无需计算"
                per_class_f1_dict[idx] = None
            elif idx not in per_class_count or per_class_count[idx] == 0:
                # 该标签在验证集中不存在样本，标记为 MISSING
                per_class_f1_dict[idx] = None
            else:
                per_class_f1_dict[idx] = float(f1)

        # 计算 macro F1、召回率、精确率（只计算可预测且存在样本的标签）
        # 获取可预测且存在样本的标签索引
        valid_labels = [
            idx for idx in range(num_classes) 
            if idx not in unpredictable_indices and per_class_count.get(idx, 0) > 0
        ]
        
        if valid_labels:
            macro_f1 = f1_score(targets_np, preds_np, labels=valid_labels, average='macro', zero_division=0)
            recall = recall_score(targets_np, preds_np, labels=valid_labels, average='macro', zero_division=0)
            precision = precision_score(targets_np, preds_np, labels=valid_labels, average='macro', zero_division=0)
        else:
            macro_f1 = 0.0
            recall = 0.0
            precision = 0.0

        metrics[task_name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "recall": recall,
            "precision": precision,
            "count": len(targets_np),
            "per_class_f1": per_class_f1_dict,
            "per_class_count": per_class_count,
            "unpredictable_labels": unpredictable_labels,
            "label_names": label_names,
            "hamming_loss": 0.0
        }
    return metrics


class FocalLoss(nn.Module):
    """
    多标签 Focal Loss，支持 per-class pos_weight。
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma: 聚焦参数，越大则越关注难分类样本（默认2.0）
    - alpha: 正类权重（类似 pos_weight），用于平衡正负样本
    - 当模型对某样本预测置信度很高（易分类）时，(1-p_t)^gamma 趋近0，
      该样本的loss贡献被大幅压低，从而避免过度鼓励模型往正方向走。
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        # gamma 可以是标量或 per-class 向量
        if isinstance(gamma, (int, float)):
            self.gamma = gamma
            self._gamma_vec = None
        else:
            self.gamma = None
            self._gamma_vec = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(gamma, dtype=torch.float)
        self.alpha = alpha  # per-class pos_weight, shape (num_classes,) or None
        self.reduction = reduction


class MLMFocalLoss(nn.Module):
    """
    适用于MLM任务的Focal Loss（单标签多分类）
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma: 聚焦参数，越大则越关注难分类样本（默认2.0）
    - alpha: per-class 权重，用于平衡类别不平衡（可选）
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape: (num_classes,)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C) 模型输出的未归一化logits
            targets: (N,) 目标类别索引
        Returns:
            focal_loss: 标量
        """
        # 计算 CrossEntropyLoss 的未归一化形式
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # (N,)
        
        # p_t = exp(-ce_loss) 对于正确类别
        pt = torch.exp(-ce_loss)
        
        # Focal Loss 调制因子
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用 focal factor
        focal_loss = focal_weight * ce_loss
        
        # 应用 per-class alpha 权重（如果有）
        if self.alpha is not None:
            # 安全索引：确保 targets 不超出 alpha 的范围
            # alpha 的长度应该等于 logits 的维度（num_classes）
            alpha_weights = self.alpha[targets.clamp(0, self.alpha.size(0) - 1)]
            focal_loss = alpha_weights * focal_loss
        
        return focal_loss.mean()


class MultiLabelFocalLoss(nn.Module):
    """
    适用于多标签分类任务的Focal Loss
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma: 聚焦参数，越大则越关注难分类样本（默认2.0）
    - alpha: per-class 权重，用于平衡类别不平衡（可选）
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape: (num_classes,)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C) 模型输出的未归一化logits
            targets: (N, C) 多标签目标（0或1）
        Returns:
            focal_loss: 标量
        """
        # 使用 sigmoid 计算概率
        probs = torch.sigmoid(logits)
        
        # 对于正样本：p_t = probs
        # 对于负样本：p_t = 1 - probs
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Focal Loss 调制因子
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算 BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 应用 focal factor
        focal_loss = focal_weight * bce_loss
        
        # 应用 per-class alpha 权重（如果有）
        if self.alpha is not None:
            # alpha 用于正样本
            alpha_weights = targets * self.alpha.unsqueeze(0) + (1 - targets) * 1.0
            focal_loss = alpha_weights * focal_loss
        
        return focal_loss.mean()


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


def compute_per_class_gamma(pos_counts, num_samples, gamma_base=1.5, gamma_max=3.0):
    """
    计算per-class gamma：小样本类别用更大的gamma，更聚焦难样本。

    策略：gamma = gamma_base + (gamma_max - gamma_base) * imbalance_ratio
    其中 imbalance_ratio = log(1 + num_samples / pos_count) / log(1 + num_samples)
    归一化到 [0, 1]，小样本类别 imbalance_ratio 接近1，大样本接近0

    Args:
        pos_counts: np.ndarray, 每个类别的正样本数
        num_samples: int, 总训练样本数
        gamma_base: float, 大样本类别的基础gamma
        gamma_max: float, 小样本类别的最大gamma

    Returns:
        torch.Tensor: per-class gamma
    """
    gammas = []
    for c in pos_counts:
        n = max(c, 1)
        # 不平衡比：小样本->接近1，大样本->接近0
        imbalance_ratio = math.log(1.0 + num_samples / n) / math.log(1.0 + num_samples)
        gamma = gamma_base + (gamma_max - gamma_base) * imbalance_ratio
        gammas.append(gamma)
    return torch.tensor(gammas, dtype=torch.float)


def parse_label(label_str):
    vec = np.zeros(len(LABEL_CLASSES), dtype=np.float32)
    if "_" in label_str:
        parts = label_str.split("_")
    else:
        parts = [label_str]
    for p in parts:
        if p in LABEL_CLASSES:
            vec[LABEL_CLASSES.index(p)] = 1.0
    return torch.tensor(vec)


def find_reverse_edges(edge_index, mask):
    """
    高效查找需要掩码的反向边（O(n)复杂度）
    
    Args:
        edge_index: (2, n) 的边索引张量
        mask: (n,) 的布尔张量，表示哪些边需要掩码
    
    Returns:
        reverse_edge_mask: (n,) 的布尔张量，表示反向边的位置
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # 将边索引存储到字典中（CPU操作，哈希查找更快）
    edge_set = {}
    for i in range(num_edges):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        edge_set[(src, dst)] = i
    
    # 查找反向边
    reverse_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    masked_indices = mask.nonzero(as_tuple=True)[0]
    
    for i in masked_indices:
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        reverse_idx = edge_set.get((dst, src))
        if reverse_idx is not None:
            reverse_edge_mask[reverse_idx] = True
    
    return reverse_edge_mask


def build_hetero_graph(sample, token_to_idx):
    graph = sample["graph"]
    metadata = graph["metadata"]

    big_nodes = graph["big_nodes"]
    small_nodes = graph["small_nodes"]
    big_edges = graph["big_edges"]
    small_edges = graph["small_edges"]

    num_big = len(big_nodes)
    num_small = len(small_nodes)

    carbon_count_list = []
    anomeric_pos_list = []
    anomeric_config_list = []
    # 记录哪些 big_node 特征不应该被掩码（包含 ? 或 x）
    big_node_maskable = []
    # 记录哪些 big_node 特征可以用于预测损失计算
    big_node_predictable = []
    # 记录每个 big_node 是否成环（0=未成环，1=成环）
    big_node_is_cyclic = []
    # 每个糖环存储所有 ring_bond 候选的 (type_idx, probability) 列表
    ring_bond_candidates_list = []
    # 记录哪些 ring_bond 不应该被掩码
    rb_maskable_list = []
    # 每个糖环存储所有 double_bond 候选的 type_idx 列表
    double_bond_candidates_list = []
    # 记录哪些 double_bond 不应该被掩码
    db_maskable_list = []

    for bn in big_nodes:
        # 获取原始特征值用于判断是否可掩码
        carbon_count_val = bn["carbon_count_type"]
        anomeric_pos_val = bn.get("anomeric_position_type", "")
        anomeric_config_val = bn.get("anomeric_config_type", "")
        
        carbon_count_list.append(token_to_idx["carbon_count_type"].get(
            carbon_count_val, token_to_idx["carbon_count_type"].get("unknown", 0)))
        anomeric_pos_list.append(token_to_idx["anomeric_position_type"].get(
            anomeric_pos_val, token_to_idx["anomeric_position_type"].get("unknown", 0)))
        anomeric_config_list.append(token_to_idx["anomeric_config_type"].get(
            anomeric_config_val, token_to_idx["anomeric_config_type"].get("unknown", 0)))
        
        # 判断该 big_node 是否可掩码（所有节点都可以掩码）
        big_node_maskable.append(1)
        
        # 判断该 big_node 是否可预测
        # carbon_count只有unknown不预测，anomeric_config的x和N/A不预测
        # N/A是"没有"，不参与预测但可以被掩码
        is_unknown_carbon = (carbon_count_val == "unknown")
        is_x_config = (anomeric_config_val == "x")
        is_na_config = (anomeric_config_val == "N/A")
        big_node_predictable.append(0 if (is_unknown_carbon or is_x_config or is_na_config) else 1)

        # 获取 ring_bonds（用于 is_cyclic 判断和后续处理）
        rbs = bn.get("ring_bonds", [])
        
        # 判断该 big_node 是否成环（is_cyclic）
        # 如果 ring_bonds 为空或只有 N/A，则 is_cyclic = 0（未成环）
        # 否则 is_cyclic = 1（成环）
        is_cyclic_val = 0
        if rbs:
            # 检查是否有非N/A的ring_bond
            has_non_na_rb = any(rb.get("ring_bond_type", "") != "N/A" for rb in rbs)
            if has_non_na_rb:
                is_cyclic_val = 1
        big_node_is_cyclic.append(is_cyclic_val)

        candidates = []
        rb_maskable = []
        
        for rb in rbs:
            rb_type = rb["ring_bond_type"]
            type_idx = token_to_idx["ring_bond_type"].get(
                rb_type, token_to_idx["ring_bond_type"].get("unknown", 0))
            prob = rb.get("probability", 1.0)
            # 记录是否带问号
            has_q = "?" in rb_type
            candidates.append((type_idx, prob, has_q))
            # 判断 ring_bond 是否可掩码
            # 不包含 ? 且概率为1.0（N/A也可以被掩码，但不参与预测）
            rb_has_question = "?" in rb_type
            rb_is_prob_one = (prob == 1.0)
            rb_maskable.append(1 if (not rb_has_question and rb_is_prob_one) else 0)
        if not candidates:
            # 没有ring_bond信息时，用N/A类型，表示没有成环
            na_idx = token_to_idx["ring_bond_type"].get("N/A", 0)
            candidates.append((na_idx, 1.0, False))
            rb_maskable.append(1)  # N/A可以被掩码，但不参与预测任务
        ring_bond_candidates_list.append(candidates)
        rb_maskable_list.append(rb_maskable)

        # 提取 double_bonds 特征
        dbs = bn.get("double_bonds", ["none"])
        db_candidates = []
        db_maskable = []
        for db in dbs:
            type_idx = token_to_idx["double_bonds"].get(
                db, token_to_idx["double_bonds"].get("none", 0))
            db_candidates.append(type_idx)
            # double_bonds 通常不包含 ? 或 x，默认可掩码
            # none表示"没有双键"，是一种有效信息，可以掩码
            db_maskable.append(1)
        if not db_candidates:
            db_candidates.append(token_to_idx["double_bonds"].get("none", 0))
            db_maskable.append(1)  # none类型可以掩码（表示"没有双键"是有效信息）
        double_bond_candidates_list.append(db_candidates)
        db_maskable_list.append(db_maskable)

    big_x = torch.stack([
        torch.tensor(carbon_count_list, dtype=torch.long),
        torch.tensor(anomeric_pos_list, dtype=torch.long),
        torch.tensor(anomeric_config_list, dtype=torch.long),
    ], dim=1)

    # 构建概率加权平均的 ring_bond embedding
    # 将所有候选展平为 (total_candidates,) 的 type_idx 和 prob，
    # 同时记录每个糖环对应的候选偏移
    rb_type_indices = []
    rb_probs = []
    rb_node_ptr = [0]  # 每个糖环在展平列表中的起始位置
    rb_has_question = []  # 记录每个 ring_bond 是否带问号
    for candidates in ring_bond_candidates_list:
        for type_idx, prob, has_q in candidates:
            rb_type_indices.append(type_idx)
            rb_probs.append(prob)
            rb_has_question.append(has_q)
        rb_node_ptr.append(rb_node_ptr[-1] + len(candidates))

    rb_type_indices_t = torch.tensor(rb_type_indices, dtype=torch.long)
    rb_probs_t = torch.tensor(rb_probs, dtype=torch.float)
    rb_node_ptr_t = torch.tensor(rb_node_ptr, dtype=torch.long)
    rb_has_question_t = torch.tensor(rb_has_question, dtype=torch.bool)

    # 构建算术平均的 double_bond embedding
    # 将所有候选展平为 (total_candidates,) 的 type_idx，
    # 同时记录每个糖环对应的候选偏移
    db_type_indices = []
    db_node_ptr = [0]  # 每个糖环在展平列表中的起始位置
    for db_candidates in double_bond_candidates_list:
        for type_idx in db_candidates:
            db_type_indices.append(type_idx)
        db_node_ptr.append(db_node_ptr[-1] + len(db_candidates))

    db_type_indices_t = torch.tensor(db_type_indices, dtype=torch.long)
    db_node_ptr_t = torch.tensor(db_node_ptr, dtype=torch.long)

    substituent_list = []
    small_node_maskable = []
    # 所有 substituent 都可被连携掩码（取代基不会有 "?"）

    for sn_idx, sn in enumerate(small_nodes):
        substituent_val = sn["substituent"]
        substituent_list.append(token_to_idx["substituent"].get(
            substituent_val, token_to_idx["substituent"].get("O", 0)))
        small_node_maskable.append(1)  # 所有 substituent 都可掩码
    small_x = torch.tensor(substituent_list, dtype=torch.long).unsqueeze(-1)
    small_node_maskable_t = torch.tensor(small_node_maskable, dtype=torch.float).unsqueeze(-1)

    # 先构建节点邻居关系，用于计算每条边能否生成负样本
    node_neighbors = {}
    existing_edges_set = set()
    for be in big_edges:
        src = be["source_big_node_id"]
        tgt = be["target_big_node_id"]
        existing_edges_set.add((src, tgt))
        if src not in node_neighbors:
            node_neighbors[src] = set()
        if tgt not in node_neighbors:
            node_neighbors[tgt] = set()
        node_neighbors[src].add(tgt)
        node_neighbors[tgt].add(src)

    # 预计算每条边的负样本生成信息
    big_edge_src_can_neg = []  # 源端能否生成负样本
    big_edge_tgt_can_neg = []  # 目标端能否生成负样本
    for be in big_edges:
        src = be["source_big_node_id"]
        tgt = be["target_big_node_id"]
        src_neighbors = node_neighbors.get(src, set())
        tgt_neighbors = node_neighbors.get(tgt, set())
        
        # 检查A能否生成假边（A的邻居中是否有与B没有边的节点）
        src_can_generate = False
        for neighbor in src_neighbors:
            if neighbor != tgt and (src, neighbor) not in existing_edges_set:
                src_can_generate = True
                break
        
        # 检查B能否生成假边（B的邻居中是否有与A没有边的节点）
        tgt_can_generate = False
        for neighbor in tgt_neighbors:
            if neighbor != src and (tgt, neighbor) not in existing_edges_set:
                tgt_can_generate = True
                break
        
        big_edge_src_can_neg.append(1 if src_can_generate else 0)
        big_edge_tgt_can_neg.append(1 if tgt_can_generate else 0)

    big_edge_index = [[], []]
    big_edge_type_list = []
    big_edge_prob_list = []
    big_edge_maskable = []
    big_edge_predictable = []  # 是否可用于损失计算
    for be in big_edges:
        src = be["source_big_node_id"]
        tgt = be["target_big_node_id"]
        big_edge_index[0].append(src)
        big_edge_index[1].append(tgt)
        big_edge_type_val = be["big_edge_type"]
        big_edge_type_list.append(token_to_idx["big_edge_type"].get(
            big_edge_type_val, 0))
        prob = be.get("probability", 1.0)
        big_edge_prob_list.append(prob)
        # 判断 big_edge 是否可掩码（所有边都可以掩码）
        big_edge_maskable.append(1)
        # 判断 big_edge 是否可用于预测（不包含 ? 且概率为1.0）
        has_question = "?" in big_edge_type_val
        is_prob_one = (prob == 1.0)
        big_edge_predictable.append(1 if (not has_question and is_prob_one) else 0)

    if big_edge_index[0]:
        big_edge_index_t = torch.tensor(big_edge_index, dtype=torch.long)
        big_edge_attr_t = torch.tensor(big_edge_type_list, dtype=torch.long).unsqueeze(-1)
        big_edge_prob_t = torch.tensor(big_edge_prob_list, dtype=torch.float).unsqueeze(-1)
        big_edge_maskable_t = torch.tensor(big_edge_maskable, dtype=torch.float).unsqueeze(-1)
        big_edge_predictable_t = torch.tensor(big_edge_predictable, dtype=torch.float).unsqueeze(-1)
        big_edge_src_can_neg_t = torch.tensor(big_edge_src_can_neg, dtype=torch.float).unsqueeze(-1)
        big_edge_tgt_can_neg_t = torch.tensor(big_edge_tgt_can_neg, dtype=torch.float).unsqueeze(-1)
        # 只有至少一个端点能生成假边，这条边才能被选为正样本
        big_edge_can_generate_neg_t = ((big_edge_src_can_neg_t + big_edge_tgt_can_neg_t) > 0).float()
    else:
        big_edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        big_edge_attr_t = torch.zeros((0, 1), dtype=torch.long)
        big_edge_prob_t = torch.zeros((0, 1), dtype=torch.float)
        big_edge_maskable_t = torch.zeros((0, 1), dtype=torch.float)
        big_edge_predictable_t = torch.zeros((0, 1), dtype=torch.float)
        big_edge_can_generate_neg_t = torch.zeros((0, 1), dtype=torch.float)
        big_edge_src_can_neg_t = torch.zeros((0, 1), dtype=torch.float)
        big_edge_tgt_can_neg_t = torch.zeros((0, 1), dtype=torch.float)

    small_edge_index = [[], []]
    stereo_type_list = []
    carbon_pos_list = []
    small_edge_prob_list = []
    small_edge_maskable = []
    small_edge_predictable = []  # 是否可用于损失计算
    for se in small_edges:
        small_edge_index[0].append(se["big_node_id"])
        small_edge_index[1].append(se["small_node_id"])
        stereo_type_val = se["stereo_type"]
        stereo_type_list.append(token_to_idx["stereo_type"].get(
            stereo_type_val, token_to_idx["stereo_type"].get("x", 0)))
        carbon_pos_list.append(token_to_idx["carbon_position_type"].get(
            se["carbon_position_type"], 0))
        prob = se.get("probability", 1.0)
        small_edge_prob_list.append(prob)
        # 判断 small_edge 是否可掩码（所有边都可以掩码）
        small_edge_maskable.append(1)
        # 判断 small_edge 是否可用于预测（不包含 ? 且不是小写 x 且概率为1.0）
        has_question = "?" in stereo_type_val
        is_x = stereo_type_val == "x"
        is_prob_one = (prob == 1.0)
        small_edge_predictable.append(1 if (not has_question and not is_x and is_prob_one) else 0)

    if small_edge_index[0]:
        small_edge_index_t = torch.tensor(small_edge_index, dtype=torch.long)
        small_edge_attr_t = torch.stack([
            torch.tensor(stereo_type_list, dtype=torch.long),
            torch.tensor(carbon_pos_list, dtype=torch.long),
        ], dim=1)
        small_edge_prob_t = torch.tensor(small_edge_prob_list, dtype=torch.float).unsqueeze(-1)
        small_edge_maskable_t = torch.tensor(small_edge_maskable, dtype=torch.float).unsqueeze(-1)
        small_edge_predictable_t = torch.tensor(small_edge_predictable, dtype=torch.float).unsqueeze(-1)
    else:
        small_edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        small_edge_attr_t = torch.zeros((0, 2), dtype=torch.long)
        small_edge_prob_t = torch.zeros((0, 1), dtype=torch.float)
        small_edge_maskable_t = torch.zeros((0, 1), dtype=torch.float)
        small_edge_predictable_t = torch.zeros((0, 1), dtype=torch.float)

    # 处理 rb_maskable 和 db_maskable
    rb_maskable = []
    for maskable in rb_maskable_list:
        rb_maskable.extend(maskable)
    rb_maskable_t = torch.tensor(rb_maskable, dtype=torch.float)
    
    db_maskable = []
    for maskable in db_maskable_list:
        db_maskable.extend(maskable)
    db_maskable_t = torch.tensor(db_maskable, dtype=torch.float)
    
    big_edge_maskable_t = torch.tensor(big_edge_maskable, dtype=torch.float).unsqueeze(-1)
    big_node_maskable_t = torch.tensor(big_node_maskable, dtype=torch.float).unsqueeze(-1)
    big_node_predictable_t = torch.tensor(big_node_predictable, dtype=torch.float).unsqueeze(-1)
    big_node_is_cyclic_t = torch.tensor(big_node_is_cyclic, dtype=torch.float).unsqueeze(-1)

    hetero_data = HeteroData()

    hetero_data["big_node"].x = big_x
    hetero_data["big_node"].maskable = big_node_maskable_t
    hetero_data["big_node"].predictable = big_node_predictable_t
    hetero_data["big_node"].is_cyclic = big_node_is_cyclic_t
    hetero_data["small_node"].x = small_x
    hetero_data["small_node"].maskable = small_node_maskable_t

    hetero_data["big_node", "big_edge", "big_node"].edge_index = big_edge_index_t
    hetero_data["big_node", "big_edge", "big_node"].edge_attr = big_edge_attr_t
    hetero_data["big_node", "big_edge", "big_node"].edge_prob = big_edge_prob_t
    hetero_data["big_node", "big_edge", "big_node"].maskable = big_edge_maskable_t
    hetero_data["big_node", "big_edge", "big_node"].predictable = big_edge_predictable_t
    hetero_data["big_node", "big_edge", "big_node"].can_generate_neg = big_edge_can_generate_neg_t
    hetero_data["big_node", "big_edge", "big_node"].src_can_neg = big_edge_src_can_neg_t
    hetero_data["big_node", "big_edge", "big_node"].tgt_can_neg = big_edge_tgt_can_neg_t

    hetero_data["big_node", "small_edge", "small_node"].edge_index = small_edge_index_t
    hetero_data["big_node", "small_edge", "small_node"].edge_attr = small_edge_attr_t
    hetero_data["big_node", "small_edge", "small_node"].edge_prob = small_edge_prob_t
    hetero_data["big_node", "small_edge", "small_node"].maskable = small_edge_maskable_t
    hetero_data["big_node", "small_edge", "small_node"].predictable = small_edge_predictable_t

    # 反向边：取代基 -> 糖环，边索引翻转，边属性和概率与原边相同
    if small_edge_index_t.size(1) > 0:
        rev_small_edge_index_t = torch.stack([small_edge_index_t[1], small_edge_index_t[0]], dim=0)
    else:
        rev_small_edge_index_t = torch.zeros((2, 0), dtype=torch.long)
    hetero_data["small_node", "rev_small_edge", "big_node"].edge_index = rev_small_edge_index_t
    hetero_data["small_node", "rev_small_edge", "big_node"].edge_attr = small_edge_attr_t
    hetero_data["small_node", "rev_small_edge", "big_node"].edge_prob = small_edge_prob_t
    hetero_data["small_node", "rev_small_edge", "big_node"].maskable = small_edge_maskable_t
    hetero_data["small_node", "rev_small_edge", "big_node"].predictable = small_edge_predictable_t

    hetero_data.global_feat = torch.tensor([
        float(metadata.get("template_count", 0)),
        float(metadata.get("sugar_count", num_big)),
        float(metadata.get("true_linkage_count", 0)),
    ], dtype=torch.float)

    # ring_bond 概率加权平均所需的数据
    hetero_data.rb_type_indices = rb_type_indices_t
    hetero_data.rb_probs = rb_probs_t
    hetero_data.rb_node_ptr = rb_node_ptr_t
    hetero_data.rb_maskable = rb_maskable_t
    hetero_data.rb_has_question = rb_has_question_t

    # double_bond 算术平均所需的数据
    hetero_data.db_type_indices = db_type_indices_t
    hetero_data.db_node_ptr = db_node_ptr_t
    hetero_data.db_maskable = db_maskable_t

    hetero_data.big_node_batch = torch.zeros(num_big, dtype=torch.long)
    hetero_data.small_node_batch = torch.zeros(num_small, dtype=torch.long)

    return hetero_data


class GlycanDataset(Dataset):
    def __init__(self, raw_data, token_to_idx):
        self.raw_data = raw_data
        self.token_to_idx = token_to_idx
        self.graphs = []
        self.labels = []
        self.glytoucan_ids = []
        self.wurcs_list = []



        print("Building graphs...")
        failed_log_path = os.path.join(os.path.dirname(__file__), "parse_failed_samples.txt")
        failed_samples = []
        for i, sample in enumerate(raw_data):
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{len(raw_data)} samples")
            try:
                g = build_hetero_graph(sample, token_to_idx)
                if g is None:
                    raise ValueError("build_hetero_graph returned None")
                label = parse_label(sample["label"])
                if label is None:
                    raise ValueError("parse_label returned None")
                self.graphs.append(g)
                self.labels.append(label)
                self.glytoucan_ids.append(sample.get("glytoucan_id", f"unknown_{i}"))
                self.wurcs_list.append(sample.get("WURCS", sample.get("wurcs", "")))
            except Exception as e:
                gid = sample.get("glytoucan_id", f"unknown_{i}")
                failed_samples.append(f"Index={i}, glytoucan_id={gid}, error={type(e).__name__}: {e}")

        if failed_samples:
            with open(failed_log_path, "w", encoding="utf-8") as f:
                f.write(f"Total failed: {len(failed_samples)} / {len(raw_data)}\n\n")
                for line in failed_samples:
                    f.write(line + "\n")
            print(f"WARNING: {len(failed_samples)} samples failed to parse, saved to {failed_log_path}")

        print(f"Done. Total: {len(self.graphs)} graphs (skipped {len(failed_samples)} failures)")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.glytoucan_ids[idx], self.wurcs_list[idx]


class DynamicBatchSampler:
    def __init__(self, indices, sizes, target_size_per_batch=500, shuffle=True):
        self.base_indices = list(indices)
        self.sizes = sizes  # dict: idx -> int (节点数+边数之和)
        self.target_size = target_size_per_batch
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        pairs = sorted(
            [(i, self.sizes[i]) for i in self.base_indices],
            key=lambda x: x[1]
        )
        batches = []
        cur_batch, cur_size = [], 0
        for idx, size in pairs:
            if cur_size + size > self.target_size and cur_batch:
                batches.append(cur_batch)
                cur_batch, cur_size = [], 0
            cur_batch.append(idx)
            cur_size += size
        if cur_batch:
            batches.append(cur_batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        pairs = sorted(
            [(i, self.sizes[i]) for i in self.base_indices],
            key=lambda x: x[1]
        )
        count = 0
        cur_size = 0
        for idx, size in pairs:
            if cur_size + size > self.target_size:
                count += 1
                cur_size = 0
            cur_size += size
        if cur_size:
            count += 1
        return count


def collate_fn(batch):
    graphs, labels, ids, wurcs = zip(*batch)

    batched_graph = HeteroData()

    for node_type in ["big_node", "small_node"]:
        xs = []
        batches = []
        maskables = []
        predictables = []
        is_cyclics = []  # is_cyclic 标记（仅 big_node 需要）
        for i, g in enumerate(graphs):
            xs.append(g[node_type].x)
            batches.append(torch.full((g[node_type].x.size(0),), i, dtype=torch.long))
            maskables.append(g[node_type].maskable)
            # 获取predictable标记，如果不存在则默认为1
            if hasattr(g[node_type], 'predictable'):
                predictables.append(g[node_type].predictable)
            else:
                predictables.append(torch.ones(g[node_type].x.size(0), 1, dtype=torch.float))
            # 获取is_cyclic标记（仅 big_node 需要）
            if node_type == "big_node":
                if hasattr(g[node_type], 'is_cyclic'):
                    is_cyclics.append(g[node_type].is_cyclic)
                else:
                    is_cyclics.append(torch.zeros(g[node_type].x.size(0), 1, dtype=torch.float))
        batched_graph[node_type].x = torch.cat(xs, dim=0)
        batched_graph[node_type].batch = torch.cat(batches, dim=0)
        batched_graph[node_type].maskable = torch.cat(maskables, dim=0)
        batched_graph[node_type].predictable = torch.cat(predictables, dim=0)
        # 仅 big_node 添加 is_cyclic
        if node_type == "big_node":
            batched_graph[node_type].is_cyclic = torch.cat(is_cyclics, dim=0)

    for edge_type_key in [("big_node", "big_edge", "big_node"),
                          ("big_node", "small_edge", "small_node"),
                          ("small_node", "rev_small_edge", "big_node")]:
        if edge_type_key in graphs[0].edge_types:
            edge_indices = []
            edge_attrs = []
            edge_probs = []
            edge_maskables = []
            edge_predictables = []  # 可预测标记
            edge_can_generate_negs = []  # 能否生成负样本标记
            edge_src_can_negs = []  # 源端能否生成负样本
            edge_tgt_can_negs = []  # 目标端能否生成负样本

            src_type = edge_type_key[0]
            dst_type = edge_type_key[2]

            src_offset = 0
            dst_offset = 0
            for i, g in enumerate(graphs):
                ei = g[edge_type_key].edge_index.clone()
                if ei.size(1) > 0:
                    ei[0] += src_offset
                    ei[1] += dst_offset
                    edge_indices.append(ei)
                    edge_attrs.append(g[edge_type_key].edge_attr)
                    edge_probs.append(g[edge_type_key].edge_prob)
                    edge_maskables.append(g[edge_type_key].maskable)
                    # 获取predictable标记，如果不存在则默认为1（全都可预测）
                    if hasattr(g[edge_type_key], 'predictable'):
                        edge_predictables.append(g[edge_type_key].predictable)
                    else:
                        edge_predictables.append(torch.ones(ei.size(1), 1, dtype=torch.float))
                    # 获取can_generate_neg标记，如果不存在则填充默认值（全0，表示不能生成负样本）
                    if hasattr(g[edge_type_key], 'can_generate_neg'):
                        edge_can_generate_negs.append(g[edge_type_key].can_generate_neg)
                    else:
                        edge_can_generate_negs.append(torch.zeros(ei.size(1), 1, dtype=torch.float))
                    # 获取src_can_neg标记，如果不存在则填充默认值（全0）
                    if hasattr(g[edge_type_key], 'src_can_neg'):
                        edge_src_can_negs.append(g[edge_type_key].src_can_neg)
                    else:
                        edge_src_can_negs.append(torch.zeros(ei.size(1), 1, dtype=torch.float))
                    # 获取tgt_can_neg标记，如果不存在则填充默认值（全0）
                    if hasattr(g[edge_type_key], 'tgt_can_neg'):
                        edge_tgt_can_negs.append(g[edge_type_key].tgt_can_neg)
                    else:
                        edge_tgt_can_negs.append(torch.zeros(ei.size(1), 1, dtype=torch.float))
                src_offset += g[src_type].x.size(0)
                dst_offset += g[dst_type].x.size(0)

            if edge_indices:
                batched_graph[edge_type_key].edge_index = torch.cat(edge_indices, dim=1)
                batched_graph[edge_type_key].edge_attr = torch.cat(edge_attrs, dim=0)
                batched_graph[edge_type_key].edge_prob = torch.cat(edge_probs, dim=0)
                batched_graph[edge_type_key].maskable = torch.cat(edge_maskables, dim=0)
                batched_graph[edge_type_key].predictable = torch.cat(edge_predictables, dim=0)
                # 合并 can_generate_neg 相关属性（所有图都有值，直接合并）
                batched_graph[edge_type_key].can_generate_neg = torch.cat(edge_can_generate_negs, dim=0)
                batched_graph[edge_type_key].src_can_neg = torch.cat(edge_src_can_negs, dim=0)
                batched_graph[edge_type_key].tgt_can_neg = torch.cat(edge_tgt_can_negs, dim=0)
            else:
                batched_graph[edge_type_key].edge_index = torch.zeros((2, 0), dtype=torch.long)
                batched_graph[edge_type_key].edge_attr = torch.zeros((0, graphs[0][edge_type_key].edge_attr.size(1)), dtype=torch.long)
                batched_graph[edge_type_key].edge_prob = torch.zeros((0, 1), dtype=torch.float)
                batched_graph[edge_type_key].maskable = torch.zeros((0, 1), dtype=torch.float)
                batched_graph[edge_type_key].predictable = torch.zeros((0, 1), dtype=torch.float)
                batched_graph[edge_type_key].can_generate_neg = torch.zeros((0, 1), dtype=torch.float)
                batched_graph[edge_type_key].src_can_neg = torch.zeros((0, 1), dtype=torch.float)
                batched_graph[edge_type_key].tgt_can_neg = torch.zeros((0, 1), dtype=torch.float)
            continue
            
        # 如果没有该边类型，初始化空数据
        batched_graph[edge_type_key].edge_index = torch.zeros((2, 0), dtype=torch.long)
        batched_graph[edge_type_key].edge_attr = torch.zeros((0, 1), dtype=torch.long)
        batched_graph[edge_type_key].edge_prob = torch.zeros((0, 1), dtype=torch.float)
        batched_graph[edge_type_key].maskable = torch.zeros((0, 1), dtype=torch.float)
        batched_graph[edge_type_key].predictable = torch.zeros((0, 1), dtype=torch.float)
        batched_graph[edge_type_key].can_generate_neg = torch.zeros((0, 1), dtype=torch.float)
        batched_graph[edge_type_key].src_can_neg = torch.zeros((0, 1), dtype=torch.float)
        batched_graph[edge_type_key].tgt_can_neg = torch.zeros((0, 1), dtype=torch.float)

    # 合并 ring_bond 概率加权平均数据
    rb_type_indices_list = []
    rb_probs_list = []
    rb_maskable_list = []
    rb_has_question_list = []
    rb_node_ptr_parts = []  # 最终拼接的所有ptr段
    rb_candidate_offset = 0
    for i, g in enumerate(graphs):
        rb_type_indices_list.append(g.rb_type_indices)
        rb_probs_list.append(g.rb_probs)
        rb_maskable_list.append(g.rb_maskable)
        rb_has_question_list.append(g.rb_has_question)
        ptr = g.rb_node_ptr.clone()
        if i == 0:
            # 第一个图：完整保留ptr，包括开头的0
            rb_node_ptr_parts.append(ptr + rb_candidate_offset)
        else:
            # 后续图：去掉开头的0（和前一个图ptr末尾重复），再偏移
            rb_node_ptr_parts.append(ptr[1:] + rb_candidate_offset)
        rb_candidate_offset += ptr[-1].item()

    batched_graph.rb_type_indices = torch.cat(rb_type_indices_list, dim=0)
    batched_graph.rb_probs = torch.cat(rb_probs_list, dim=0)
    batched_graph.rb_maskable = torch.cat(rb_maskable_list, dim=0)
    batched_graph.rb_has_question = torch.cat(rb_has_question_list, dim=0)
    batched_graph.rb_node_ptr = torch.cat(rb_node_ptr_parts, dim=0)

    # 合并 double_bond 算术平均数据
    db_type_indices_list = []
    db_maskable_list = []
    db_node_ptr_parts = []
    db_candidate_offset = 0
    for i, g in enumerate(graphs):
        db_type_indices_list.append(g.db_type_indices)
        db_maskable_list.append(g.db_maskable)
        ptr = g.db_node_ptr.clone()
        if i == 0:
            db_node_ptr_parts.append(ptr + db_candidate_offset)
        else:
            db_node_ptr_parts.append(ptr[1:] + db_candidate_offset)
        db_candidate_offset += ptr[-1].item()

    batched_graph.db_type_indices = torch.cat(db_type_indices_list, dim=0)
    batched_graph.db_maskable = torch.cat(db_maskable_list, dim=0)
    batched_graph.db_node_ptr = torch.cat(db_node_ptr_parts, dim=0)

    global_feats = torch.stack([g.global_feat for g in graphs], dim=0)

    labels_batch = torch.stack(labels, dim=0)

    return batched_graph, global_feats, labels_batch, list(ids), list(wurcs)


class HeteroGNN(nn.Module):
    def __init__(self, vocab_sizes, hidden_dim=128, num_layers=3, dropout=0.2, num_classes=None,
                 num_heads=4, mlm_probability=0.15, freq_stats_path=None, token_to_idx=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_heads = num_heads
        self.mlm_probability = mlm_probability
        # GAT输出维度 = hidden_dim，每头维度 = hidden_dim // num_heads
        self.head_dim = hidden_dim // num_heads
        self.token_to_idx = token_to_idx  # 用于动态掩码概率计算和获取MASK token索引
        
        # 加载频率统计，用于动态调整掩码概率
        self.freq_stats = {}
        self.high_freq_tokens = {}
        if freq_stats_path and os.path.exists(freq_stats_path):
            self._load_freq_stats(freq_stats_path)
            # 标记高频特征（出现频次超过总频次的10%视为高频）
            self._identify_high_freq_tokens()

        # ---- 节点嵌入（自适应维度：根据词汇表大小分配嵌入维度）----
        # 经验法则：embed_dim ≈ 4 * ceil(log2(vocab_size))
        self.big_emb_dims = {
            "carbon_count_type":       12,   # vocab=7, 4*ceil(log2(7))=4*3=12
            "ring_bond_type":          16,   # vocab=11, 4*ceil(log2(11))=4*4=16
            "anomeric_position_type":   8,   # vocab=4, 4*ceil(log2(4))=4*2=8（作为输入，但不预测）
            "anomeric_config_type":     8,   # vocab=4, 4*ceil(log2(4))=4*2=8
            "double_bonds":             4,   # vocab=2, 4*ceil(log2(2))=4*1=4
        }
        self.big_embeddings = nn.ModuleDict({
            k: nn.Embedding(vocab_sizes[k], d) for k, d in self.big_emb_dims.items()
        })
        # Concat + 投影替代逐个投影后相加，保留各特征独立信息
        big_concat_dim = sum(self.big_emb_dims.values())  # 12+16+8+8+4 = 48
        self.big_emb_fuse = nn.Sequential(
            nn.Linear(big_concat_dim, hidden_dim),
            nn.ReLU(),
        )

        self.small_emb_dim = 32  # 增加维度以更好地学习复杂的取代基特征
        self.small_embeddings = nn.ModuleDict({
            "substituent": nn.Embedding(vocab_sizes["substituent"], self.small_emb_dim),
        })
        self.small_emb_proj = nn.Linear(self.small_emb_dim, hidden_dim)

        # ---- 边嵌入 ----
        self.big_edge_emb_dim = 24  # vocab=35, 4*ceil(log2(35))=4*6=24
        self.big_edge_emb = nn.Embedding(vocab_sizes["big_edge_type"], self.big_edge_emb_dim)
        self.big_edge_proj = nn.Linear(self.big_edge_emb_dim, hidden_dim)

        self.small_edge_stereo_dim = 16  # vocab=15, 4*ceil(log2(15))=4*4=16
        self.small_edge_stereo_emb = nn.Embedding(vocab_sizes["stereo_type"], self.small_edge_stereo_dim)

        self.small_edge_carbon_dim = 16  # vocab=9, 4*ceil(log2(9))=4*4=16
        self.small_edge_carbon_emb = nn.Embedding(vocab_sizes["carbon_position_type"], self.small_edge_carbon_dim)

        # 拼接 stereo(16) + carbon(16) = 32，再统一投影到 hidden_dim
        self.small_edge_fuse_proj = nn.Linear(self.small_edge_stereo_dim + self.small_edge_carbon_dim, hidden_dim)
        
        # ---- MLM 预测头 ----
        # 用于预测被掩码的特征
        self.mlm_heads = nn.ModuleDict({
            "carbon_count_type": nn.Linear(hidden_dim, vocab_sizes["carbon_count_type"]),
            # "anomeric_position_type": nn.Linear(hidden_dim, vocab_sizes["anomeric_position_type"]),  # 已删除
            "anomeric_config_type": nn.Linear(hidden_dim, vocab_sizes["anomeric_config_type"]),
            "ring_bond_type": nn.Linear(hidden_dim, vocab_sizes["ring_bond_type"]),
            "double_bonds": nn.Linear(hidden_dim, vocab_sizes["double_bonds"]),
            "substituent": nn.Linear(hidden_dim, vocab_sizes["substituent"]),
            "big_edge_type": nn.Linear(hidden_dim * 2, vocab_sizes["big_edge_type"]),  # 拼接两个节点嵌入
            "stereo_type": nn.Linear(hidden_dim * 2, vocab_sizes["stereo_type"]),        # 拼接两个节点嵌入
            "carbon_position_type": nn.Linear(hidden_dim, vocab_sizes["carbon_position_type"]),
            "is_cyclic": nn.Linear(hidden_dim, 2),  # 新增：是否成环二分类预测
        })
        
        # ---- 边存在性预测头 ----
        # 用于预测边是否存在（二元分类）
        self.edge_exists_heads = nn.ModuleDict({
            "big_edge_exists": nn.Linear(hidden_dim * 2, 2),  # 拼接两个节点嵌入
            "small_edge_exists": nn.Linear(hidden_dim * 2, 2),  # 拼接两个节点嵌入
        })

        # ---- _fuse_edge_features 投影层（GAT之前的概率加权边特征聚合）----
        self.big_edge_fuse = nn.Linear(hidden_dim, hidden_dim)
        self.small_edge_fuse = nn.Linear(hidden_dim, hidden_dim)
        self.rev_small_edge_fuse = nn.Linear(hidden_dim, hidden_dim)

        # ---- 边概率投影层（将edge_prob融入GAT边特征）----
        # big_edge: hidden_dim + 1(prob) -> hidden_dim
        self.big_edge_prob_proj = nn.Linear(hidden_dim + 1, hidden_dim)
        # small_edge: hidden_dim + 1(prob) -> hidden_dim
        self.small_edge_prob_proj = nn.Linear(hidden_dim + 1, hidden_dim)


        # ---- GAT卷积层（节点级注意力 + 边特征感知）----
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # 糖环-糖环：同质GAT，边特征参与注意力计算
                ("big_node", "big_edge", "big_node"): GATConv(
                    hidden_dim, hidden_dim, heads=num_heads,
                    concat=False, dropout=dropout, add_self_loops=True,
                    edge_dim=hidden_dim
                ),
                # 糖环-取代基：异质GAT，边特征参与注意力计算
                ("big_node", "small_edge", "small_node"): GATConv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=num_heads,
                    concat=False, dropout=dropout, add_self_loops=False,
                    edge_dim=hidden_dim
                ),
                # 反向边：取代基-糖环，边特征参与注意力计算
                ("small_node", "rev_small_edge", "big_node"): GATConv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=num_heads,
                    concat=False, dropout=dropout, add_self_loops=False,
                    edge_dim=hidden_dim
                ),
            }, aggr="sum")
            self.convs.append(conv)

        self.norm_big = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm_small = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])


    def _fuse_edge_features(self, x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict):
        """概率加权边特征聚合：在GAT之前增强节点表示"""
        big_enhanced = x_dict["big_node"]
        small_enhanced = x_dict["small_node"]

        big_edge_idx = ("big_node", "big_edge", "big_node")
        if big_edge_idx in edge_attr_dict and edge_attr_dict[big_edge_idx].size(0) > 0:
            be_emb = self.big_edge_proj(self.big_edge_emb(edge_attr_dict[big_edge_idx][:, 0]))
            be_prob = edge_prob_dict.get(big_edge_idx,
                                         torch.ones(be_emb.size(0), 1, device=be_emb.device))
            be_weighted = be_emb * be_prob
            # 确保数据类型一致（混合精度训练时可能需要）
            be_weighted = be_weighted.to(big_enhanced.dtype)
            src_be = edge_index_dict[big_edge_idx][0]
            dst_be = edge_index_dict[big_edge_idx][1]
            big_src = torch.zeros_like(big_enhanced)
            big_dst = torch.zeros_like(big_enhanced)
            big_src.scatter_add_(0, src_be.unsqueeze(-1).expand(-1, big_enhanced.size(1)),
                                 be_weighted)
            big_dst.scatter_add_(0, dst_be.unsqueeze(-1).expand(-1, big_enhanced.size(1)),
                                 be_weighted)
            big_enhanced = big_enhanced + self.big_edge_fuse(big_src + big_dst)

        small_edge_idx = ("big_node", "small_edge", "small_node")
        if small_edge_idx in edge_attr_dict and edge_attr_dict[small_edge_idx].size(0) > 0:
            se_attr = edge_attr_dict[small_edge_idx]
            se_emb = self.small_edge_fuse_proj(torch.cat([
                self.small_edge_stereo_emb(se_attr[:, 0]),
                self.small_edge_carbon_emb(se_attr[:, 1]),
            ], dim=-1))
            se_prob = edge_prob_dict.get(small_edge_idx,
                                         torch.ones(se_emb.size(0), 1, device=se_emb.device))
            se_weighted = se_emb * se_prob
            # 确保数据类型一致（混合精度训练时可能需要）
            se_weighted = se_weighted.to(x_dict["small_node"].dtype)
            dst_se = edge_index_dict[small_edge_idx][1]
            small_aggregated = torch.zeros_like(x_dict["small_node"])
            small_aggregated.scatter_add_(0, dst_se.unsqueeze(-1).expand(-1, small_aggregated.size(1)),
                                          se_weighted)
            small_enhanced = small_enhanced + self.small_edge_fuse(small_aggregated)

        rev_small_edge_idx = ("small_node", "rev_small_edge", "big_node")
        if rev_small_edge_idx in edge_attr_dict and edge_attr_dict[rev_small_edge_idx].size(0) > 0:
            rse_attr = edge_attr_dict[rev_small_edge_idx]
            rse_emb = self.small_edge_fuse_proj(torch.cat([
                self.small_edge_stereo_emb(rse_attr[:, 0]),
                self.small_edge_carbon_emb(rse_attr[:, 1]),
            ], dim=-1))
            rse_prob = edge_prob_dict.get(rev_small_edge_idx,
                                          torch.ones(rse_emb.size(0), 1, device=rse_emb.device))
            rse_weighted = rse_emb * rse_prob
            # 确保数据类型一致（混合精度训练时可能需要）
            rse_weighted = rse_weighted.to(big_enhanced.dtype)
            dst_rse = edge_index_dict[rev_small_edge_idx][1]
            big_from_small = torch.zeros_like(big_enhanced)
            big_from_small.scatter_add_(0, dst_rse.unsqueeze(-1).expand(-1, big_enhanced.size(1)),
                                        rse_weighted)
            big_enhanced = big_enhanced + self.rev_small_edge_fuse(big_from_small)

        return {"big_node": big_enhanced, "small_node": small_enhanced}


    def forward(self, x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                batch_dict, global_feat, batched_graph=None, return_mlm_logits=False):
        big_x = x_dict["big_node"]
        small_x = x_dict["small_node"]
        
        # 保存原始标签用于MLM损失计算
        original_big_x = big_x.clone()
        original_small_x = small_x.clone()
        original_rb_indices = batched_graph.rb_type_indices.clone()
        original_db_indices = batched_graph.db_type_indices.clone()
        original_big_edge_attr = None
        original_small_edge_attr = None
        original_big_edge_predictable = None  # 保存big_edge的predictable标记
        original_small_edge_predictable = None  # 保存small_edge的predictable标记
        if ("big_node", "big_edge", "big_node") in edge_attr_dict:
            original_big_edge_attr = edge_attr_dict[("big_node", "big_edge", "big_node")].clone()
            if hasattr(batched_graph[("big_node", "big_edge", "big_node")], 'predictable'):
                original_big_edge_predictable = batched_graph[("big_node", "big_edge", "big_node")].predictable.squeeze(-1).bool()
            else:
                original_big_edge_predictable = torch.ones(original_big_edge_attr.size(0), dtype=torch.bool, device=original_big_edge_attr.device)
        if ("big_node", "small_edge", "small_node") in edge_attr_dict:
            original_small_edge_attr = edge_attr_dict[("big_node", "small_edge", "small_node")].clone()
            if hasattr(batched_graph[("big_node", "small_edge", "small_node")], 'predictable'):
                original_small_edge_predictable = batched_graph[("big_node", "small_edge", "small_node")].predictable.squeeze(-1).bool()
            else:
                original_small_edge_predictable = torch.ones(original_small_edge_attr.size(0), dtype=torch.bool, device=original_small_edge_attr.device)

        # 获取边的负样本生成能力（用于边存在性预测任务）
        edge_can_generate_neg_dict = {et: batched_graph[et].can_generate_neg if hasattr(batched_graph[et], 'can_generate_neg') else None 
                                       for et in batched_graph.edge_types}
        edge_src_can_neg_dict = {et: batched_graph[et].src_can_neg if hasattr(batched_graph[et], 'src_can_neg') else None 
                                  for et in batched_graph.edge_types}
        edge_tgt_can_neg_dict = {et: batched_graph[et].tgt_can_neg if hasattr(batched_graph[et], 'tgt_can_neg') else None 
                                  for et in batched_graph.edge_types}

        # ---- 掩码操作 ----
        # 获取maskable标记
        big_node_maskable = batched_graph["big_node"].maskable if hasattr(batched_graph["big_node"], 'maskable') else torch.ones(big_x.size(0), 1, device=big_x.device)
        small_node_maskable = batched_graph["small_node"].maskable if hasattr(batched_graph["small_node"], 'maskable') else torch.ones(small_x.size(0), 1, device=small_x.device)
        
        # 记录被掩码的big_node索引（用于后续移除连接的边）
        masked_big_nodes = torch.zeros(big_x.size(0), dtype=torch.bool, device=big_x.device)
        
        # 保存原始边信息（用于计算损失）
        original_small_edge_index = None
        if ("big_node", "small_edge", "small_node") in edge_index_dict:
            original_small_edge_index = edge_index_dict[("big_node", "small_edge", "small_node")].clone()
        
        # 对可掩码的big_node特征进行掩码（支持动态掩码概率）
        # 训练和验证时都需要掩码（验证时也需要计算MLM损失）
        if self.training or return_mlm_logits:
            # 【is_cyclic任务】构建均衡的训练样本
            # 策略：选择15%的N/A样本（未成环）和相同数量的正常样本（成环）
            is_cyclic_mask = torch.zeros(big_x.size(0), dtype=torch.bool, device=big_x.device)
            
            # 获取 is_cyclic 标签（0=未成环/N/A, 1=成环）
            is_cyclic_labels = batched_graph["big_node"].is_cyclic.squeeze(-1)
            
            # 找到 N/A 节点（未成环）和正常节点（成环）
            na_nodes = torch.where(is_cyclic_labels == 0)[0]
            cyclic_nodes = torch.where(is_cyclic_labels == 1)[0]
            
            # 选择15%的N/A节点
            na_sample_prob = 0.15
            num_na_to_select = max(1, int(len(na_nodes) * na_sample_prob))
            
            if len(na_nodes) > 0:
                # 随机选择N/A节点
                na_selected = na_nodes[torch.randperm(len(na_nodes), device=big_x.device)[:num_na_to_select]]
                is_cyclic_mask[na_selected] = True
            
            # 选择相同数量的正常节点（成环）
            num_cyclic_to_select = min(num_na_to_select, len(cyclic_nodes))
            if len(cyclic_nodes) > 0 and num_cyclic_to_select > 0:
                cyclic_selected = cyclic_nodes[torch.randperm(len(cyclic_nodes), device=big_x.device)[:num_cyclic_to_select]]
                is_cyclic_mask[cyclic_selected] = True
            
            # big_node: carbon_count_type (col 0), anomeric_position (col 1), anomeric_config (col 2)
            # carbon_count_type 可以掩码，但需要同时移除连接的边防止数据泄露
            # anomeric_config_type 参与随机掩码
            # anomeric_position_type 参与随机掩码，但不预测（无需计算损失）
            categories = ["carbon_count_type", "anomeric_position_type", "anomeric_config_type"]
            cols_to_mask = [0, 1, 2]  # 掩码碳数、异头碳位置和异头碳构型
            
            for col, category in zip(cols_to_mask, categories):
                mask_prob = torch.rand(big_x.size(0), device=big_x.device)
                # 获取每个token的掩码概率（高频特征概率降低）
                token_indices = big_x[:, col]
                
                # 碳数预测任务使用较低的掩码概率（10%），其他任务使用默认概率
                if col == 0:  # carbon_count_type
                    dynamic_probs = torch.ones_like(mask_prob) * 0.10  # 下调至10%
                else:
                    dynamic_probs = torch.ones_like(mask_prob) * self.mlm_probability
                
                if category in self.freq_stats and self.token_to_idx is not None:
                    for token_idx in token_indices.unique():
                        prob = self._get_mask_probability(category, token_idx.item())
                        dynamic_probs[token_indices == token_idx] = prob
                
                # 计算可掩码的项目数（基于整个batch）
                maskable_items = (big_node_maskable[:, 0] == 1)
                
                # 与is_cyclic任务互斥处理
                if col == 0:
                    # carbon_count: 完全排除 is_cyclic_mask 的节点
                    maskable_items = maskable_items & (~is_cyclic_mask)
                else:
                    # anomeric_position/anomeric_config: 对 is_cyclic_mask 的节点强制掩码
                    pass
                
                num_maskable = int(maskable_items.sum().item())
                num_to_mask = max(1, int(num_maskable * self.mlm_probability))
                
                # 生成随机掩码概率并选择
                mask = torch.zeros(big_x.size(0), dtype=torch.bool, device=big_x.device)
                if num_maskable > 0:
                    maskable_indices = torch.where(maskable_items)[0]
                    num_to_select = min(num_to_mask, len(maskable_indices))
                    if num_to_select > 0:
                        permuted_indices = torch.randperm(len(maskable_indices), device=big_x.device)[:num_to_select]
                        selected_indices = maskable_indices[permuted_indices]
                        mask[selected_indices] = True
                
                # 对 is_cyclic_mask 的节点强制掩码（仅对 anomeric_position/anomeric_config）
                if col != 0:
                    is_cyclic_force_mask = is_cyclic_mask & (big_node_maskable[:, 0] == 1)
                    mask = mask | is_cyclic_force_mask
                
                # 使用正确的MASK token索引
                mask_idx = get_mask_token_idx(self.token_to_idx, category) if self.token_to_idx else len(self.big_embeddings[category].weight)
                big_x[mask, col] = mask_idx
                
                # 如果是碳数掩码，记录被掩码的big_node
                if col == 0:  # carbon_count_type
                    masked_big_nodes |= mask
                
                # 如果是异头碳构型掩码，记录被掩码的 N/A（用于连携掩码 ring_bond）
                # 但 is_cyclic_mask 的节点不参与这个连携掩码逻辑（它们有自己的任务）
                if col == 2:  # anomeric_config_type
                    pass  # anomeric_config N/A 的连携掩码逻辑已在 is_cyclic 任务中处理
            
            # 当掩码碳数时，移除连接的 small_edge（防止通过边连接推断碳数）
            if masked_big_nodes.any():
                
                small_edge_key = ("big_node", "small_edge", "small_node")
                rev_small_edge_key = ("small_node", "rev_small_edge", "big_node")
                
                if small_edge_key in edge_index_dict and edge_index_dict[small_edge_key].size(1) > 0:
                    small_edge_index = edge_index_dict[small_edge_key]
                    small_edge_attr = edge_attr_dict.get(small_edge_key)
                    small_edge_prob = edge_prob_dict.get(small_edge_key)
                    
                    # 找到不连接到被掩码big_node的边
                    src_big_nodes = small_edge_index[0]
                    edges_to_keep = ~masked_big_nodes[src_big_nodes]
                    
                    # 更新边索引、属性和概率
                    edge_index_dict[small_edge_key] = small_edge_index[:, edges_to_keep]
                    if small_edge_attr is not None:
                        edge_attr_dict[small_edge_key] = small_edge_attr[edges_to_keep]
                    if small_edge_prob is not None:
                        edge_prob_dict[small_edge_key] = small_edge_prob[edges_to_keep]
                    
                    # 同时更新反向边
                    if rev_small_edge_key in edge_index_dict:
                        edge_index_dict[rev_small_edge_key] = edge_index_dict[rev_small_edge_key][:, edges_to_keep]
                        if rev_small_edge_key in edge_attr_dict and edge_attr_dict[rev_small_edge_key] is not None:
                            edge_attr_dict[rev_small_edge_key] = edge_attr_dict[rev_small_edge_key][edges_to_keep]
                        if rev_small_edge_key in edge_prob_dict and edge_prob_dict[rev_small_edge_key] is not None:
                            edge_prob_dict[rev_small_edge_key] = edge_prob_dict[rev_small_edge_key][edges_to_keep]
                    
                    # 更新 original_small_edge_attr 和 predictable 以匹配过滤后的边（用于 small_edge_stereo 的 MLM 损失计算）
                    if original_small_edge_attr is not None:
                        original_small_edge_attr = original_small_edge_attr[edges_to_keep]
                    if original_small_edge_predictable is not None:
                        original_small_edge_predictable = original_small_edge_predictable[edges_to_keep]
            
            # small_node: substituent（高频特征如"O"降低掩码概率）
            # 计算可掩码的项目数（基于整个batch）
            maskable_items = (small_node_maskable[:, 0] == 1)
            num_maskable = int(maskable_items.sum().item())
            num_to_mask = max(1, int(num_maskable * self.mlm_probability))
            
            # 生成随机掩码
            mask = torch.zeros(small_x.size(0), dtype=torch.bool, device=small_x.device)
            if num_maskable > 0:
                maskable_indices = torch.where(maskable_items)[0]
                num_to_select = min(num_to_mask, len(maskable_indices))
                if num_to_select > 0:
                    permuted_indices = torch.randperm(len(maskable_indices), device=small_x.device)[:num_to_select]
                    selected_indices = maskable_indices[permuted_indices]
                    mask[selected_indices] = True
            
            # 使用正确的MASK token索引
            mask_idx = get_mask_token_idx(self.token_to_idx, "substituent") if self.token_to_idx else len(self.small_embeddings["substituent"].weight)
            small_x[mask, 0] = mask_idx

        # ring_bond_type 概率加权平均嵌入（在原始嵌入空间聚合，再拼接投影）
        rb_type_indices = batched_graph.rb_type_indices
        rb_probs = batched_graph.rb_probs
        rb_node_ptr = batched_graph.rb_node_ptr
        rb_maskable = batched_graph.rb_maskable if hasattr(batched_graph, 'rb_maskable') else torch.ones(rb_type_indices.size(0), device=rb_type_indices.device)
        
        # 获取节点数量（用于联合掩码）
        num_big_nodes = big_x.size(0)
        
        # 对ring_bond进行掩码（支持动态掩码概率）
        # 训练和验证时都需要掩码（验证时也需要计算MLM损失）
        if self.training or return_mlm_logits:
            total_rb_candidates = rb_type_indices.size(0)
            
            # 使用 rb_node_ptr 构建 ring_bond 到节点的映射
            rb_to_node = torch.zeros(total_rb_candidates, dtype=torch.long, device=rb_type_indices.device)
            for i in range(num_big_nodes):
                start = rb_node_ptr[i].item()
                end = rb_node_ptr[i+1].item() if i+1 < len(rb_node_ptr) else total_rb_candidates
                rb_to_node[start:end] = i
            
            # 【is_cyclic任务】使用前面已定义的 is_cyclic_mask（在 big_node 特征掩码部分定义）
            # 对选中的节点进行 ring_bond_type 掩码（anomeric_position 和 anomeric_config 已经在前面掩码了）
            rb_masked_big_nodes_for_cyclic = is_cyclic_mask.clone()
            
            if is_cyclic_mask.any():
                # 掩码 ring_bond_type（其他两个特征已经在前面掩码了）
                rb_type_indices = rb_type_indices.clone()
                rb_mask_idx = get_mask_token_idx(self.token_to_idx, "ring_bond_type") if self.token_to_idx else len(self.big_embeddings["ring_bond_type"].weight)
                
                # 找到选中节点对应的所有 ring_bond candidates
                for node_idx in torch.where(is_cyclic_mask)[0]:
                    start = rb_node_ptr[node_idx].item()
                    end = rb_node_ptr[node_idx+1].item() if node_idx+1 < len(rb_node_ptr) else total_rb_candidates
                    rb_type_indices[start:end] = rb_mask_idx
            
            # 【普通ring_bond掩码】对非is_cyclic任务的节点进行常规掩码
            # 排除已被is_cyclic任务选中的节点
            # 新策略：以节点为单位进行掩码，选中节点时掩码所有确定环信息（概率=1.0）
            # 关键规则：
            # 1. 排除带概率环的节点（有任何ring_bond概率<1.0的节点都不能参与）
            # 2. 只保留所有ring_bond概率都>=1.0的节点（纯确定环节点）
            
            # 首先找出哪些节点是纯确定环节点（所有ring_bond概率都>=1.0）
            pure_deterministic_nodes = torch.zeros(num_big_nodes, dtype=torch.bool, device=rb_type_indices.device)
            for i in range(num_big_nodes):
                start = rb_node_ptr[i].item()
                end = rb_node_ptr[i+1].item() if i+1 < len(rb_node_ptr) else total_rb_candidates
                node_rb_probs = rb_probs[start:end]
                
                # 只有当所有ring_bond概率都>=1.0时，才是纯确定环节点
                # 排除有任何概率<1.0的节点（包括一个确定环配一个不确定环的情况）
                all_deterministic = (node_rb_probs >= 1.0).all() if len(node_rb_probs) > 0 else False
                if all_deterministic:
                    pure_deterministic_nodes[i] = True
            
            # 排除已被is_cyclic任务选中的节点
            maskable_nodes_for_rb = pure_deterministic_nodes & (~is_cyclic_mask)
            
            # 统计整个batch中可掩码的节点数
            num_maskable = int(maskable_nodes_for_rb.sum().item())
            
            # 确定要掩码的节点数（基于整个batch的可掩码节点数）
            num_to_mask = max(1, int(num_maskable * self.mlm_probability))
            
            # 对可掩码节点进行随机选择
            selected_nodes_for_rb_mask = torch.zeros(num_big_nodes, dtype=torch.bool, device=rb_type_indices.device)
            if num_maskable > 0:
                # 获取可掩码节点的索引
                maskable_indices = torch.where(maskable_nodes_for_rb)[0]
                
                # 随机选择指定数量的节点
                num_to_select = min(num_to_mask, len(maskable_indices))
                if num_to_select > 0:
                    permuted_indices = torch.randperm(len(maskable_indices), device=rb_type_indices.device)[:num_to_select]
                    selected_indices = maskable_indices[permuted_indices]
                    selected_nodes_for_rb_mask[selected_indices] = True
            
            # 对选中节点的所有确定环信息进行掩码
            mask = torch.zeros(total_rb_candidates, dtype=torch.bool, device=rb_type_indices.device)
            if selected_nodes_for_rb_mask.any():
                rb_type_indices = rb_type_indices.clone()
                rb_mask_idx = get_mask_token_idx(self.token_to_idx, "ring_bond_type") if self.token_to_idx else len(self.big_embeddings["ring_bond_type"].weight)
                
                for node_idx in torch.where(selected_nodes_for_rb_mask)[0]:
                    start = rb_node_ptr[node_idx].item()
                    end = rb_node_ptr[node_idx+1].item() if node_idx+1 < len(rb_node_ptr) else total_rb_candidates
                    rb_type_indices[start:end] = rb_mask_idx
                    mask[start:end] = True
            
            # 记录随机掩码的 ring_bond 对应的 big_node（用于连携掩码 anomeric_position）
            masked_rb_big_nodes = torch.zeros(num_big_nodes, dtype=torch.bool, device=rb_type_indices.device)
            if mask.any():
                masked_rb_big_nodes[rb_to_node[mask]] = True
            
            # 当普通ring_bond被掩码时，对应的 anomeric_position 也应该掩码
            # 排除已被is_cyclic任务选中的节点
            masked_rb_big_nodes = masked_rb_big_nodes & (~is_cyclic_mask)
            if masked_rb_big_nodes.any():
                ap_mask_idx = get_mask_token_idx(self.token_to_idx, "anomeric_position_type") if self.token_to_idx else len(self.big_embeddings["anomeric_position_type"].weight)
                big_x[masked_rb_big_nodes, 1] = ap_mask_idx

        rb_all_emb = self.big_embeddings["ring_bond_type"](rb_type_indices)  # (total_candidates, emb_dim=24)
        rb_all_weighted = rb_all_emb * rb_probs.unsqueeze(-1)  # 概率加权

        # 用 segment reduce 按节点聚合：每个节点的加权 embedding 求和，再除以概率和
        rb_emb_sum = torch.zeros(num_big_nodes, rb_all_weighted.size(1), device=rb_all_weighted.device)
        rb_prob_sum = torch.zeros(num_big_nodes, device=rb_all_weighted.device)

        # 优化：使用纯张量操作替代 for 循环，大幅提升计算速度
        candidate_counts = rb_node_ptr[1:] - rb_node_ptr[:-1]
        candidate_node_ids = torch.arange(num_big_nodes, device=rb_all_weighted.device).repeat_interleave(candidate_counts)

        rb_emb_sum.scatter_add_(0, candidate_node_ids.unsqueeze(-1).expand(-1, rb_all_weighted.size(1)), rb_all_weighted)
        rb_prob_sum.scatter_add_(0, candidate_node_ids, rb_probs)

        # 加权平均 = 加权和 / 概率和
        rb_emb = rb_emb_sum / rb_prob_sum.unsqueeze(-1).clamp(min=1e-8)  # (num_big_nodes, 24)

        # double_bond_type 算术平均嵌入（在原始嵌入空间聚合，再拼接投影）
        db_type_indices = batched_graph.db_type_indices
        db_node_ptr = batched_graph.db_node_ptr
        db_maskable = batched_graph.db_maskable if hasattr(batched_graph, 'db_maskable') else torch.ones(db_type_indices.size(0), device=db_type_indices.device)
        
        # 对double_bond进行掩码
        # 训练和验证时都需要掩码（验证时也需要计算MLM损失）
        if self.training or return_mlm_logits:
            mask_prob = torch.rand(db_type_indices.size(0), device=db_type_indices.device)
            mask = (mask_prob < self.mlm_probability) & (db_maskable == 1)
            db_type_indices = db_type_indices.clone()
            db_type_indices[mask] = 0

        db_all_emb = self.big_embeddings["double_bonds"](db_type_indices)  # (total_candidates, emb_dim=8)

        # ---- 边特征掩码 ----
        big_edge_key = ("big_node", "big_edge", "big_node")
        small_edge_key = ("big_node", "small_edge", "small_node")
        rev_small_edge_key = ("small_node", "rev_small_edge", "big_node")
        
        # 边存在性预测：随机删除一部分big_edge，记录被删除的边
        deleted_big_edge_mask = None
        original_big_edge_index = None
        original_big_edge_predictable = None  # 保存原始边的predictable标记用于损失计算
        
        # 对big_edge_type进行掩码
        if big_edge_key in edge_attr_dict and edge_attr_dict[big_edge_key].size(0) > 0:
            num_edges = edge_attr_dict[big_edge_key].size(0)
            big_edge_maskable = batched_graph[big_edge_key].maskable if hasattr(batched_graph[big_edge_key], 'maskable') else torch.ones(num_edges, 1, device=edge_attr_dict[big_edge_key].device)
            # 保存predictable标记用于后续损失计算
            if hasattr(batched_graph[big_edge_key], 'predictable'):
                original_big_edge_predictable = batched_graph[big_edge_key].predictable.squeeze(-1).bool()
            else:
                original_big_edge_predictable = torch.ones(num_edges, dtype=torch.bool, device=edge_attr_dict[big_edge_key].device)
            
            # 训练和验证时都需要掩码（验证时也需要计算MLM损失）
            if self.training or return_mlm_logits:
                # 计算可掩码的项目数（基于整个batch）
                maskable_items = (big_edge_maskable[:, 0] == 1)
                num_maskable = int(maskable_items.sum().item())
                num_to_mask = max(1, int(num_maskable * self.mlm_probability))
                
                # 生成随机掩码
                mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_attr_dict[big_edge_key].device)
                if num_maskable > 0:
                    maskable_indices = torch.where(maskable_items)[0]
                    num_to_select = min(num_to_mask, len(maskable_indices))
                    if num_to_select > 0:
                        permuted_indices = torch.randperm(len(maskable_indices), device=edge_attr_dict[big_edge_key].device)[:num_to_select]
                        selected_indices = maskable_indices[permuted_indices]
                        mask[selected_indices] = True
                
                # 找到反向边并一起掩码（防止数据泄露）
                # 使用 O(n) 哈希表查找替代 O(n²) 循环
                edge_index = edge_index_dict[big_edge_key]
                reverse_edge_mask = find_reverse_edges(edge_index, mask)
                
                # 同时掩码正向边和反向边
                combined_mask = mask | reverse_edge_mask
                
                edge_attr_dict[big_edge_key] = edge_attr_dict[big_edge_key].clone()
                # 使用正确的MASK token索引
                mask_idx = get_mask_token_idx(self.token_to_idx, "big_edge_type") if self.token_to_idx else len(self.big_edge_emb.weight)
                edge_attr_dict[big_edge_key][combined_mask, 0] = mask_idx
        
        # 边存在性预测：随机删除一部分big_edge用于预测
        # 保存原始边索引用于计算损失
        deleted_big_edge_mask = None
        original_big_edge_index = None
        negative_big_edge_index = None
        
        if big_edge_key in edge_index_dict and edge_index_dict[big_edge_key].size(1) > 0:
            original_big_edge_index = edge_index_dict[big_edge_key].clone()
            num_edges = original_big_edge_index.size(1)
            num_big_nodes = x_dict["big_node"].size(0) if "big_node" in x_dict else 0
            
            # 获取边的概率信息
            big_edge_probs = edge_prob_dict.get(big_edge_key)
            
            # 检查条件：是否有足够的确定边作为正样本（至少2条）
            can_do_edge_existence = False
            if big_edge_probs is not None and big_edge_probs.size(0) > 0:
                num_certain_edges = (big_edge_probs >= 1.0).sum().item()
                if num_certain_edges >= 2:
                    can_do_edge_existence = True
            
            if self.training or return_mlm_logits:
                if can_do_edge_existence:
                    # 只选择 probability=1.0 的确定边
                    certain_edge_mask = (big_edge_probs >= 1.0)
                    
                    # 基于确定边随机选择约15%进行删除
                    num_certain_edges = certain_edge_mask.sum().item()
                    target_delete_count = int(num_certain_edges * self.mlm_probability * 0.5)
                    
                    if target_delete_count > 0:
                        # 随机选择要删除的边
                        certain_indices = certain_edge_mask.nonzero(as_tuple=True)[0]
                        perm = torch.randperm(certain_indices.size(0), device=certain_indices.device)
                        selected_indices = certain_indices[perm[:target_delete_count]]
                        
                        # 创建删除掩码
                        deleted_big_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=original_big_edge_index.device)
                        deleted_big_edge_mask[selected_indices] = True
                        
                        # 找到反向边并一起删除（防止数据泄露，因为大边的反向边是包含在数据中显式送入的）
                        edge_index = edge_index_dict[big_edge_key]
                        reverse_edge_mask = find_reverse_edges(edge_index, deleted_big_edge_mask)
                        
                        # 同时删除正向边和反向边
                        combined_delete_mask = deleted_big_edge_mask | reverse_edge_mask
                        
                        # 删除被选中的边及其反向边
                        edge_index_dict[big_edge_key] = original_big_edge_index[:, ~combined_delete_mask]
                        
                        # 同时删除对应的边属性和概率
                        if big_edge_key in edge_attr_dict and edge_attr_dict[big_edge_key] is not None:
                            edge_attr_dict[big_edge_key] = edge_attr_dict[big_edge_key][~combined_delete_mask]
                        if big_edge_key in edge_prob_dict and edge_prob_dict[big_edge_key] is not None:
                            edge_prob_dict[big_edge_key] = edge_prob_dict[big_edge_key][~combined_delete_mask]
                        
                        # 更新 original_big_edge_attr 和 predictable 以匹配过滤后的边
                        if original_big_edge_attr is not None:
                            original_big_edge_attr = original_big_edge_attr[~combined_delete_mask]
                        if original_big_edge_predictable is not None:
                            original_big_edge_predictable = original_big_edge_predictable[~combined_delete_mask]
                        
                        # 生成负样本
                        if num_big_nodes > 0:
                            deleted_edge_indices = deleted_big_edge_mask.nonzero(as_tuple=True)[0]
                            deleted_src_nodes = original_big_edge_index[0, deleted_edge_indices]
                            deleted_tgt_nodes = original_big_edge_index[1, deleted_edge_indices]
                            
                            # 收集已存在的边
                            existing_edges_set = set()
                            for i in range(original_big_edge_index.size(1)):
                                s = original_big_edge_index[0, i].item()
                                t = original_big_edge_index[1, i].item()
                                existing_edges_set.add((s, t))
                            
                            # 收集邻居信息
                            node_neighbors = {}
                            for s, t in existing_edges_set:
                                if s not in node_neighbors:
                                    node_neighbors[s] = set()
                                if t not in node_neighbors:
                                    node_neighbors[t] = set()
                                node_neighbors[s].add(t)
                                node_neighbors[t].add(s)
                            
                            # 为每条被删除的边生成负样本
                            # 负样本策略：构造不存在的假边，用于训练模型判断边是否存在
                            neg_src = []
                            neg_tgt = []
                            for i in range(deleted_edge_indices.size(0)):
                                src = deleted_src_nodes[i].item()
                                tgt = deleted_tgt_nodes[i].item()
                                
                                # 为src生成负样本边：(src, rand_node) 不存在于图中
                                # 策略：70%从tgt的邻居中选择（更难的负样本），30%随机选择
                                tgt_neighbors = list(node_neighbors.get(tgt, set()))
                                found_neg_for_src = False
                                if tgt_neighbors and torch.rand(()).item() < 0.7:
                                    # 从tgt的邻居中筛选出不与src相连的节点
                                    valid_neighbors = [n for n in tgt_neighbors if n != src and (src, n) not in existing_edges_set]
                                    if valid_neighbors:
                                        rand_node = valid_neighbors[torch.randint(0, len(valid_neighbors), (1,)).item()]
                                        neg_src.append(src)
                                        neg_tgt.append(rand_node)
                                        found_neg_for_src = True
                                
                                if not found_neg_for_src:
                                    # 随机选择不存在的边
                                    candidates = [n for n in range(num_big_nodes) if n != src and n != tgt and (src, n) not in existing_edges_set]
                                    if candidates:
                                        rand_node = candidates[torch.randint(0, len(candidates), (1,)).item()]
                                        neg_src.append(src)
                                        neg_tgt.append(rand_node)
                                
                                # 为tgt生成负样本边：(tgt, rand_node) 不存在于图中
                                # 策略：70%从src的邻居中选择（更难的负样本），30%随机选择
                                src_neighbors = list(node_neighbors.get(src, set()))
                                found_neg_for_tgt = False
                                if src_neighbors and torch.rand(()).item() < 0.7:
                                    # 从src的邻居中筛选出不与tgt相连的节点
                                    valid_neighbors = [n for n in src_neighbors if n != tgt and (tgt, n) not in existing_edges_set]
                                    if valid_neighbors:
                                        rand_node = valid_neighbors[torch.randint(0, len(valid_neighbors), (1,)).item()]
                                        neg_src.append(tgt)
                                        neg_tgt.append(rand_node)
                                        found_neg_for_tgt = True
                                
                                if not found_neg_for_tgt:
                                    # 随机选择不存在的边
                                    candidates = [n for n in range(num_big_nodes) if n != src and n != tgt and (tgt, n) not in existing_edges_set]
                                    if candidates:
                                        rand_node = candidates[torch.randint(0, len(candidates), (1,)).item()]
                                        neg_src.append(tgt)
                                        neg_tgt.append(rand_node)
                            
                            if neg_src:
                                negative_big_edge_index = torch.stack([
                                    torch.tensor(neg_src, device=original_big_edge_index.device),
                                    torch.tensor(neg_tgt, device=original_big_edge_index.device)
                                ], dim=0)
                            else:
                                negative_big_edge_index = None
                        else:
                            negative_big_edge_index = None
                    else:
                        can_do_edge_existence = False
                
                if not can_do_edge_existence:
                    deleted_big_edge_mask = None
                    negative_big_edge_index = None
        
        # 对small_edge的stereo_type进行掩码（carbon_position_type不掩码）
        if small_edge_key in edge_attr_dict and edge_attr_dict[small_edge_key].size(0) > 0:
            # 获取small_edge_maskable，并确保它与当前的edge_attr_dict大小一致
            # （因为可能在前面的碳数掩码步骤中已经过滤了部分边）
            original_maskable = batched_graph[small_edge_key].maskable if hasattr(batched_graph[small_edge_key], 'maskable') else torch.ones(batched_graph[small_edge_key].edge_index.size(1), 1, device=edge_attr_dict[small_edge_key].device)
            
            # 如果边已经被过滤，需要相应地更新maskable
            if original_maskable.size(0) != edge_attr_dict[small_edge_key].size(0):
                # 找到不连接到被掩码big_node的边的索引
                if masked_big_nodes is not None and masked_big_nodes.any():
                    original_edge_index = batched_graph[small_edge_key].edge_index
                    src_big_nodes = original_edge_index[0]
                    edges_to_keep = ~masked_big_nodes[src_big_nodes]
                    small_edge_maskable = original_maskable[edges_to_keep]
                else:
                    small_edge_maskable = original_maskable[:edge_attr_dict[small_edge_key].size(0)]
            else:
                small_edge_maskable = original_maskable
            
            # 训练和验证时都需要掩码（验证时也需要计算MLM损失）
            if self.training or return_mlm_logits:
                # 计算可掩码的项目数（基于整个batch）
                maskable_items = (small_edge_maskable[:, 0] == 1)
                num_maskable = int(maskable_items.sum().item())
                num_to_mask = max(1, int(num_maskable * self.mlm_probability))
                
                # 生成随机掩码
                mask = torch.zeros(edge_attr_dict[small_edge_key].size(0), dtype=torch.bool, device=edge_attr_dict[small_edge_key].device)
                if num_maskable > 0:
                    maskable_indices = torch.where(maskable_items)[0]
                    num_to_select = min(num_to_mask, len(maskable_indices))
                    if num_to_select > 0:
                        permuted_indices = torch.randperm(len(maskable_indices), device=edge_attr_dict[small_edge_key].device)[:num_to_select]
                        selected_indices = maskable_indices[permuted_indices]
                        mask[selected_indices] = True
                
                # 找到反向边并一起掩码（防止数据泄露）
                # small_edge 的反向边是自动生成的（rev_small_edge），索引一一对应
                # 所以直接使用相同的 mask 作为 combined_mask
                combined_mask = mask
                
                # 使用正确的MASK token索引
                mask_idx = get_mask_token_idx(self.token_to_idx, "stereo_type") if self.token_to_idx else len(self.small_edge_stereo_emb.weight)
                
                edge_attr_dict[small_edge_key] = edge_attr_dict[small_edge_key].clone()
                edge_attr_dict[small_edge_key][combined_mask, 0] = mask_idx  # 只掩码stereo_type（第0列），carbon_position_type（第1列）不掩码
                
                # 同时掩码反向边
                if rev_small_edge_key in edge_attr_dict:
                    edge_attr_dict[rev_small_edge_key] = edge_attr_dict[rev_small_edge_key].clone()
                    edge_attr_dict[rev_small_edge_key][combined_mask, 0] = mask_idx  # 只掩码stereo_type（第0列）
                
                # 注意：不修改 edge_index，保持图结构完整
                # GAT 会使用 edge_attr 来计算注意力，掩码后的 edge_attr 会被处理为 MASK token 的 embedding
                
                # 注意：substituent 采用独立随机掩码，不与 stereo 连携掩码
                # 双键类型的连携掩码也已取消，避免过度掩码导致数据稀疏

        # 用 segment reduce 按节点聚合：每个节点的 embedding 求和，再除以候选数
        num_big_nodes = big_x.size(0)
        db_emb_sum = torch.zeros(num_big_nodes, db_all_emb.size(1), device=db_all_emb.device)

        # 使用纯张量操作替代 for 循环
        db_candidate_counts = db_node_ptr[1:] - db_node_ptr[:-1]
        db_candidate_node_ids = torch.arange(num_big_nodes, device=db_all_emb.device).repeat_interleave(db_candidate_counts)

        db_emb_sum.scatter_add_(0, db_candidate_node_ids.unsqueeze(-1).expand(-1, db_all_emb.size(1)), db_all_emb)

        # 算术平均 = 总和 / 候选数
        db_emb = db_emb_sum / db_candidate_counts.unsqueeze(-1).clamp(min=1).float()  # (num_big_nodes, 8)

        # 拼接所有大节点特征嵌入，再统一投影到 hidden_dim
        big_emb = self.big_emb_fuse(torch.cat([
            self.big_embeddings["carbon_count_type"](big_x[:, 0]),   # (N, 12)
            rb_emb,                                                     # (N, 16)
            self.big_embeddings["anomeric_position_type"](big_x[:, 1]), # (N, 8)（作为输入，但不预测）
            self.big_embeddings["anomeric_config_type"](big_x[:, 2]),   # (N, 8)
            db_emb,                                                     # (N, 4)
        ], dim=-1))  # concat: (N, 56) -> proj -> (N, hidden_dim)

        small_emb = self.small_emb_proj(self.small_embeddings["substituent"](small_x[:, 0].flatten()))
        
        # 保存用于MLM预测的中间表示
        mlm_big_emb = big_emb
        mlm_small_emb = small_emb

        x_dict_out = {
            "big_node": big_emb,
            "small_node": small_emb,
        }

        # ---- GAT消息传递（节点级注意力 + 边特征感知）----
        # 预计算边特征嵌入（所有层共享同一组边特征）
        edge_feat_dict = {}
        big_edge_key = ("big_node", "big_edge", "big_node")
        small_edge_key = ("big_node", "small_edge", "small_node")
        rev_small_edge_key = ("small_node", "rev_small_edge", "big_node")

        if big_edge_key in edge_attr_dict and edge_attr_dict[big_edge_key].size(0) > 0:
            be_emb = self.big_edge_proj(
                self.big_edge_emb(edge_attr_dict[big_edge_key][:, 0])
            )
            be_prob = edge_prob_dict.get(big_edge_key,
                                         torch.ones(be_emb.size(0), 1, device=be_emb.device))
            edge_feat_dict[big_edge_key] = self.big_edge_prob_proj(
                torch.cat([be_emb, be_prob], dim=-1)
            )
        if small_edge_key in edge_attr_dict and edge_attr_dict[small_edge_key].size(0) > 0:
            se_attr = edge_attr_dict[small_edge_key]
            se_emb = self.small_edge_fuse_proj(torch.cat([
                self.small_edge_stereo_emb(se_attr[:, 0]),
                self.small_edge_carbon_emb(se_attr[:, 1]),
            ], dim=-1))
            se_prob = edge_prob_dict.get(small_edge_key,
                                         torch.ones(se_emb.size(0), 1, device=se_emb.device))
            edge_feat_dict[small_edge_key] = self.small_edge_prob_proj(
                torch.cat([se_emb, se_prob], dim=-1)
            )
        if rev_small_edge_key in edge_attr_dict and edge_attr_dict[rev_small_edge_key].size(0) > 0:
            rse_attr = edge_attr_dict[rev_small_edge_key]
            rse_emb = self.small_edge_fuse_proj(torch.cat([
                self.small_edge_stereo_emb(rse_attr[:, 0]),
                self.small_edge_carbon_emb(rse_attr[:, 1]),
            ], dim=-1))
            rse_prob = edge_prob_dict.get(rev_small_edge_key,
                                          torch.ones(rse_emb.size(0), 1, device=rse_emb.device))
            edge_feat_dict[rev_small_edge_key] = self.small_edge_prob_proj(
                torch.cat([rse_emb, rse_prob], dim=-1)
            )

        for layer_idx, conv in enumerate(self.convs):
            x_dict_res = x_dict_out

            x_dict_out = conv(x_dict_out, edge_index_dict, edge_feat_dict)

            x_dict_out["big_node"] = self.norm_big[layer_idx](
                x_dict_out["big_node"] + x_dict_res["big_node"]
            )
            x_dict_out["big_node"] = F.relu(x_dict_out["big_node"])
            x_dict_out["big_node"] = F.dropout(x_dict_out["big_node"],
                                               p=self.dropout_rate, training=self.training)

            x_dict_out["small_node"] = self.norm_small[layer_idx](
                x_dict_out["small_node"] + x_dict_res["small_node"]
            )
            x_dict_out["small_node"] = F.relu(x_dict_out["small_node"])
            x_dict_out["small_node"] = F.dropout(x_dict_out["small_node"],
                                                 p=self.dropout_rate, training=self.training)

        # ---- 直接拼接 mean+max 池化结果 ----
        # 不使用投影层，避免预训练时无法训练的参数
        # big_node: mean(256) + max(256) = 512维
        # small_node: mean(256) + max(256) = 512维
        # 最终: 512 + 512 + 3 = 1027维
        big_mean = global_mean_pool(x_dict_out["big_node"], batch_dict["big_node"])
        big_max = global_max_pool(x_dict_out["big_node"], batch_dict["big_node"])
        big_pooled = torch.cat([big_mean, big_max], dim=-1)  # 512维

        small_mean = global_mean_pool(x_dict_out["small_node"], batch_dict["small_node"])
        small_max = global_max_pool(x_dict_out["small_node"], batch_dict["small_node"])
        small_pooled = torch.cat([small_mean, small_max], dim=-1)  # 512维

        # 直接拼接得到1024维图表示，再加3维全局特征 → 1027维
        graph_pooled = torch.cat([big_pooled, small_pooled], dim=-1)  # 1024维
        features = torch.cat([graph_pooled, global_feat], dim=-1)  # 1027维
        
        # ---- MLM 预测 ----
        if return_mlm_logits:
            # 使用最终的节点嵌入来预测被掩码的特征
            mlm_logits = {
                "carbon_count_type": self.mlm_heads["carbon_count_type"](mlm_big_emb),
                # "anomeric_position_type": self.mlm_heads["anomeric_position_type"](mlm_big_emb),  # 已删除
                "anomeric_config_type": self.mlm_heads["anomeric_config_type"](mlm_big_emb),
                "ring_bond_type": self.mlm_heads["ring_bond_type"](x_dict_out["big_node"]),
                "double_bonds": self.mlm_heads["double_bonds"](x_dict_out["big_node"]),
                "substituent": self.mlm_heads["substituent"](mlm_small_emb),
                "is_cyclic": self.mlm_heads["is_cyclic"](x_dict_out["big_node"]),  # 新增：是否成环预测
            }
            
            # 边特征的MLM预测
            big_edge_key = ("big_node", "big_edge", "big_node")
            if big_edge_key in edge_index_dict and edge_index_dict[big_edge_key].size(1) > 0:
                # 使用源节点和目标节点的嵌入拼接来预测边特征
                src_nodes = edge_index_dict[big_edge_key][0]
                tgt_nodes = edge_index_dict[big_edge_key][1]
                big_edge_emb = torch.cat([x_dict_out["big_node"][src_nodes], x_dict_out["big_node"][tgt_nodes]], dim=-1)
                mlm_logits["big_edge_type"] = self.mlm_heads["big_edge_type"](big_edge_emb)
            
            # 边存在性预测：使用被删除的边（正样本）和负样本
            if deleted_big_edge_mask is not None and deleted_big_edge_mask.any():
                # 正样本：被删除的边（应该存在）
                deleted_edge_index = original_big_edge_index[:, deleted_big_edge_mask]
                pos_src_nodes = deleted_edge_index[0]
                pos_tgt_nodes = deleted_edge_index[1]
                pos_big_edge_emb = torch.cat([x_dict_out["big_node"][pos_src_nodes], x_dict_out["big_node"][pos_tgt_nodes]], dim=-1)
                
                # 负样本：随机生成的不存在的边（不应该存在）
                if negative_big_edge_index is not None and negative_big_edge_index.size(1) > 0:
                    neg_src_nodes = negative_big_edge_index[0]
                    neg_tgt_nodes = negative_big_edge_index[1]
                    neg_big_edge_emb = torch.cat([x_dict_out["big_node"][neg_src_nodes], x_dict_out["big_node"][neg_tgt_nodes]], dim=-1)
                    
                    # 拼接正负样本
                    combined_edge_emb = torch.cat([pos_big_edge_emb, neg_big_edge_emb], dim=0)
                    mlm_logits["big_edge_exists"] = self.edge_exists_heads["big_edge_exists"](combined_edge_emb)
                else:
                    mlm_logits["big_edge_exists"] = self.edge_exists_heads["big_edge_exists"](pos_big_edge_emb)
            
            small_edge_key = ("big_node", "small_edge", "small_node")
            if small_edge_key in edge_index_dict and edge_index_dict[small_edge_key].size(1) > 0:
                # 使用源节点和目标节点的嵌入拼接来预测边特征
                src_nodes = edge_index_dict[small_edge_key][0]
                tgt_nodes = edge_index_dict[small_edge_key][1]
                small_edge_emb = torch.cat([x_dict_out["big_node"][src_nodes], x_dict_out["small_node"][tgt_nodes]], dim=-1)
                mlm_logits["stereo_type"] = self.mlm_heads["stereo_type"](small_edge_emb)
            
            # 原始标签（用于计算MLM损失）
            mlm_labels = {
                "big_x": original_big_x,
                "small_x": original_small_x,
                "rb_indices": original_rb_indices,
                "rb_masked_indices": rb_type_indices,  # 添加被掩码后的ring_bond索引
                "db_indices": original_db_indices,
                "big_edge_attr": original_big_edge_attr,
                "small_edge_attr": original_small_edge_attr,
                "big_edge_predictable": original_big_edge_predictable,
                "small_edge_predictable": original_small_edge_predictable,
                "deleted_big_edge_mask": deleted_big_edge_mask,  # 用于边存在性损失
                "negative_big_edge_index": negative_big_edge_index,  # 负样本边索引
                "is_cyclic": batched_graph["big_node"].is_cyclic,  # 添加是否成环标签
                "rb_masked_for_cyclic": rb_masked_big_nodes_for_cyclic,  # ring_bond被掩码的节点（用于is_cyclic任务）
            }
            
            # 生成 ring_bond 的多标签目标（只包含概率为1且不带问号的环键）
            num_big_nodes = big_x.size(0)
            num_rb_classes = self.mlm_heads["ring_bond_type"].out_features
            ring_bond_labels = torch.zeros(num_big_nodes, num_rb_classes, device=rb_type_indices.device)
            
            # 获取 ring_bond 是否带问号的信息
            rb_has_question = batched_graph.rb_has_question if hasattr(batched_graph, 'rb_has_question') else torch.zeros(rb_type_indices.size(0), dtype=torch.bool, device=rb_type_indices.device)
            
            # 获取N/A的索引（索引0）
            na_idx = self.token_to_idx["ring_bond_type"].get("N/A", 0) if self.token_to_idx else 0
            
            # 调试：统计生成的标签
            total_positive_samples = 0
            
            # 遍历每个节点的 ring_bond 候选
            for i in range(num_big_nodes):
                start = rb_node_ptr[i].item()
                end = rb_node_ptr[i+1].item() if i+1 < len(rb_node_ptr) else len(rb_type_indices)
                
                # 获取该节点的所有 ring_bond 候选（使用原始索引来判断N/A）
                node_rb_types = original_rb_indices[start:end]
                node_rb_probs = rb_probs[start:end]
                node_rb_has_q = rb_has_question[start:end]
                
                # 只保留概率为1且不带问号的环键
                valid_mask = (node_rb_probs >= 1.0) & (~node_rb_has_q)
                valid_types = node_rb_types[valid_mask]

                # 获取正确的 MASK token 索引
                mask_idx = get_mask_token_idx(self.token_to_idx, "ring_bond_type") if self.token_to_idx else len(self.big_embeddings["ring_bond_type"].weight)

                # 设置多标签
                has_valid_ring_bond = False
                for rb_type in valid_types:
                    if rb_type != mask_idx:  # 正确排除MASK token（索引11），而非N/A（索引0）
                        ring_bond_labels[i, rb_type] = 1.0
                        total_positive_samples += 1
                        has_valid_ring_bond = True
                
                # N/A 已被标记为不可预测标签，不参与预测，所以不生成 N/A 标签
                # 如果没有有效的环键（即只有N/A，没有成环），则不设置任何标签（保持全0）
                # 这样该样本不参与 ring_bond 任务的损失计算
            
            mlm_labels["ring_bond_labels"] = ring_bond_labels
            mlm_labels["ring_bond_probs"] = rb_probs  # 用于损失计算时过滤
            
            return features, mlm_logits, mlm_labels
        
        return features

    def _load_freq_stats(self, freq_stats_path):
        """加载频率统计文件"""
        with open(freq_stats_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = content.split('------------------------------------------------------------')
        for section in sections:
            if 'unique values' in section:
                lines = section.strip().split('\n')
                if len(lines) < 2:
                    continue
                category = lines[0].replace('[', '').replace(']', '').strip()
                if category not in self.freq_stats:
                    self.freq_stats[category] = {}
                for line in lines[1:]:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            token = parts[0].strip()
                            freq = float(parts[1].strip())
                            self.freq_stats[category][token] = freq

    def _identify_high_freq_tokens(self):
        """识别高频特征，对高频特征降低掩码概率"""
        for category, tokens in self.freq_stats.items():
            if not tokens:
                continue
            total_freq = sum(tokens.values())
            if total_freq == 0:
                continue
            # 频次超过总频次10%的视为高频特征
            self.high_freq_tokens[category] = set()
            for token, freq in tokens.items():
                if freq / total_freq > 0.1:  # 超过10%视为高频
                    self.high_freq_tokens[category].add(token)
            if self.high_freq_tokens[category]:
                print(f"High frequency tokens in {category}: {self.high_freq_tokens[category]}")

    def _get_mask_probability(self, category, token_idx):
        """根据特征频率获取动态掩码概率（基于频率衰减）
        
        策略：使用平方根衰减函数
        mask_prob = base_prob * sqrt(max(0.1, 1 - freq_ratio))
        
        这样高频特征掩码概率降低，但不会低于 base_prob * sqrt(0.1) ≈ 32% of base_prob
        """
        base_prob = self.mlm_probability
        
        # 如果没有频率统计或token_to_idx，使用基础概率
        if category not in self.freq_stats or self.token_to_idx is None:
            return base_prob
        
        # 查找token对应的字符串
        token_str = None
        for t, idx in self.token_to_idx.items():
            if idx == token_idx:
                token_str = t
                break
        
        if token_str and token_str in self.freq_stats[category]:
            # 计算该特征的频率占比
            total_freq = sum(self.freq_stats[category].values())
            freq_ratio = self.freq_stats[category][token_str] / total_freq
            
            # 使用平方根衰减：频率越高，掩码概率越低
            # max(0.1, ...) 确保最低掩码概率不会太低
            mask_prob = base_prob * math.sqrt(max(0.1, 1 - freq_ratio))
            return mask_prob
        
        return base_prob

    def _mask_with_protection(self, x, category, maskable, min_keep_ratio=0.1, min_keep_count=1):
        """带批次级保护的掩码操作
        
        Args:
            x: 特征张量
            category: 特征类别名称
            maskable: 是否可掩码的标记
            min_keep_ratio: 每个特征类别至少保留的比例（默认10%）
            min_keep_count: 每个特征类别至少保留的数量（默认1个）
        """
        if not self.training:
            return x, torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        
        x_masked = x.clone()
        mask_prob = torch.rand(x.size(0), device=x.device)
        
        # 获取token索引
        if x.dim() > 1:
            token_indices = x[:, 0]
        else:
            token_indices = x
        
        # 计算动态掩码概率
        dynamic_probs = torch.ones_like(mask_prob) * self.mlm_probability
        if category in self.freq_stats and self.token_to_idx is not None:
            for token_idx in token_indices.unique():
                prob = self._get_mask_probability(category, token_idx.item())
                dynamic_probs[token_indices == token_idx] = prob
        
        # 初始掩码
        mask = (mask_prob < dynamic_probs) & (maskable == 1)
        
        # ===== 批次级保护：确保每个特征类别至少保留一定数量的样本 =====
        for token_idx in token_indices.unique():
            token_mask = (token_indices == token_idx)
            token_count = token_mask.sum().item()
            masked_count = (token_mask & mask).sum().item()
            
            # 计算最小保留数量
            min_keep = max(min_keep_count, int(token_count * min_keep_ratio))
            max_mask = token_count - min_keep
            
            if masked_count > max_mask:
                # 需要取消部分掩码
                to_unmask = masked_count - max_mask
                masked_indices = torch.where(token_mask & mask)[0]
                if len(masked_indices) > 0:
                    # 随机选择一些位置取消掩码
                    unmask_indices = masked_indices[torch.randperm(len(masked_indices))[:to_unmask]]
                    mask[unmask_indices] = False
        
        x_masked[mask] = 0  # MASK token
        return x_masked, mask

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
    
    def close(self):
        """关闭日志文件"""
        if hasattr(self, 'log') and self.log is not None:
            self.log.close()
            self.log = None
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()



class EarlyStopping:
    """支持基于loss(越小越好)或F1(越大越好)的早停，支持EMA滑动平均"""
    def __init__(self, patience=30, min_delta=0.0, mode="min", ema_window=5, use_ema=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" for loss, "max" for f1/accuracy
        self.ema_window = ema_window
        self.use_ema = use_ema
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.value_history = []  # 存储原始值历史
        self.ema_value = None    # 当前EMA值

    def compute_ema(self):
        """计算滑动平均值"""
        if len(self.value_history) == 0:
            return None
        # 使用最近 ema_window 个值
        recent_values = self.value_history[-self.ema_window:]
        return sum(recent_values) / len(recent_values)

    def __call__(self, value):
        # 记录原始值
        self.value_history.append(value)
        
        # 计算EMA
        self.ema_value = self.compute_ema()
        
        # 决定用原始值还是EMA来判断
        judge_value = self.ema_value if self.use_ema else value
        
        if self.mode == "min":
            score = -judge_value
        else:
            score = judge_value
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


def train_one_epoch(model, loader, optimizer, scaler, max_grad_norm=1.0, task_criteria=None, vocab_sizes=None, token_to_idx=None, epoch=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 记录每个任务的损失
    task_losses = {
        # "anomeric_position": 0.0,  # 已删除，不再作为预测任务
        "carbon_count": 0.0,        # 新增：碳数预测
        "anomeric_config": 0.0,
        "ring_bond": 0.0,           # ring_bond多标签预测，只对概率为1的计算损失
        "substituent": 0.0,         # 取代基预测任务，O不参与loss计算
        "big_edge": 0.0,
        "small_edge_stereo": 0.0,
        "big_edge_exists": 0.0,     # 新增：大环边存在性预测
        "is_cyclic": 0.0,           # 新增：是否成环二分类任务
    }
    task_counts = {k: 0 for k in task_losses.keys()}

    # 收集每个任务的预测和标签，用于计算分类指标
    task_preds = {k: [] for k in task_losses.keys()}
    task_targets = {k: [] for k in task_losses.keys()}

    # 为每个任务创建带有类别平衡权重的 Focal Loss
    if task_criteria is None:
        task_criteria = {}
    mlm_criteria = {}
    for task_name in task_losses.keys():
        if task_name == "ring_bond":
            # ring_bond 是多标签任务，使用 MultiLabelFocalLoss
            if task_name in task_criteria and task_criteria[task_name] is not None:
                mlm_criteria[task_name] = MultiLabelFocalLoss(gamma=3.0, alpha=task_criteria[task_name])
            else:
                mlm_criteria[task_name] = MultiLabelFocalLoss(gamma=3.0)
        else:
            # 其他任务是单标签任务，使用 MLMFocalLoss
            if task_name in task_criteria and task_criteria[task_name] is not None:
                mlm_criteria[task_name] = MLMFocalLoss(gamma=3.0, alpha=task_criteria[task_name])
            else:
                mlm_criteria[task_name] = MLMFocalLoss(gamma=3.0)
    
    # 边存在性预测使用二元交叉熵损失
    edge_exists_criterion = nn.CrossEntropyLoss()

    for batch_idx, (batched_graph, global_feat, labels, ids, wurcs) in enumerate(loader):
        batched_graph = batched_graph.to(DEVICE)
        global_feat = global_feat.to(DEVICE)

        if batch_idx % 10 == 0 or batch_idx == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | GPU Mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

        x_dict = {nt: batched_graph[nt].x for nt in batched_graph.node_types}
        edge_index_dict = {et: batched_graph[et].edge_index for et in batched_graph.edge_types}
        edge_attr_dict = {et: batched_graph[et].edge_attr for et in batched_graph.edge_types}
        edge_prob_dict = {et: batched_graph[et].edge_prob for et in batched_graph.edge_types}
        edge_can_generate_neg_dict = {et: batched_graph[et].can_generate_neg if hasattr(batched_graph[et], 'can_generate_neg') else None 
                                       for et in batched_graph.edge_types}
        edge_src_can_neg_dict = {et: batched_graph[et].src_can_neg if hasattr(batched_graph[et], 'src_can_neg') else None 
                                  for et in batched_graph.edge_types}
        edge_tgt_can_neg_dict = {et: batched_graph[et].tgt_can_neg if hasattr(batched_graph[et], 'tgt_can_neg') else None 
                                  for et in batched_graph.edge_types}
        batch_dict = {nt: batched_graph[nt].batch for nt in batched_graph.node_types}

        optimizer.zero_grad()
        
        # 混合精度训练：使用 autocast 包装前向传播和损失计算
        with autocast('cuda'):
            # 启用MLM训练
            features, mlm_logits, mlm_labels = model(x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                                               batch_dict, global_feat, batched_graph, return_mlm_logits=True)
            
            # MLM损失
            mlm_loss = 0.0
            
            # 任务损失权重配置 - 给难学的任务更高权重
            loss_weights = {
                # "anomeric_position": 1.0,  # 已删除，不再作为预测任务
                "carbon_count": 1.0,
                "anomeric_config": 1.0,
                "ring_bond": 1.0,        # ring_bond多标签预测权重，只对概率为1的计算损失
                "substituent": 1.0,         # 取代基预测任务权重，O不参与loss计算
                "big_edge": 1.0,
                "small_edge_stereo": 1.0,
                "big_edge_exists": 1.0,
                "is_cyclic": 1.0,        # 是否成环二分类任务权重
            }
            
            # big_node特征的MLM损失（碳数和异头碳构型）
            # cols: 0=carbon_count, 1=anomeric_position(可掩码但不预测), 2=anomeric_config
            for i, (feat_name, task_name) in enumerate([("carbon_count_type", "carbon_count"), 
                                                        ("anomeric_config_type", "anomeric_config")]):
                col = [0, 2][i]  # 对应big_x的第0列（碳数）和第2列（anomeric_config）
                pred = mlm_logits[feat_name]
                target = mlm_labels["big_x"][:, col]
                # 使用正确的MASK token索引判断被掩码的位置
                mask_idx = get_mask_token_idx(token_to_idx, feat_name) if token_to_idx else len(model.big_embeddings[feat_name].weight)
                mask = (x_dict["big_node"][:, col] == mask_idx)
                # 获取predictable标记
                if hasattr(batched_graph["big_node"], 'predictable'):
                    predictable = batched_graph["big_node"].predictable.squeeze(-1).bool()
                    mask = mask & predictable
                if mask.any():
                    loss = mlm_criteria[task_name](pred[mask], target[mask])
                    mlm_loss += loss * loss_weights[task_name]
                    task_losses[task_name] += loss.item()
                    task_counts[task_name] += 1
                    # 收集预测和标签用于计算指标
                    task_preds[task_name].append(pred[mask].argmax(dim=1))
                    task_targets[task_name].append(target[mask])
            
            # small_node substituent 特征的MLM损失（O不参与loss计算）
            if "substituent" in mlm_logits:
                pred_substituent = mlm_logits["substituent"]
                target_substituent = mlm_labels["small_x"][:, 0]
                current_small_x = x_dict["small_node"][:, 0]
                # 使用正确的MASK token索引判断被掩码的位置
                mask_idx = get_mask_token_idx(token_to_idx, "substituent") if token_to_idx else len(model.small_embeddings["substituent"].weight)
                mask_substituent = (current_small_x == mask_idx)
                # 排除O（无需预测项），动态获取O的索引
                o_idx = token_to_idx["substituent"].get("O", 13) if token_to_idx else 13
                not_o_mask = (target_substituent != o_idx)
                mask_substituent = mask_substituent & not_o_mask
                if mask_substituent.any():
                    loss = mlm_criteria["substituent"](pred_substituent[mask_substituent], target_substituent[mask_substituent])
                    mlm_loss += loss * loss_weights["substituent"]
                    task_losses["substituent"] += loss.item()
                    task_counts["substituent"] += 1
                    # 收集预测和标签用于计算指标（排除O）
                    task_preds["substituent"].append(pred_substituent[mask_substituent].argmax(dim=1))
                    task_targets["substituent"].append(target_substituent[mask_substituent])
            
            # ring_bond 特征的多标签MLM损失（只对概率为1的环键计算损失）
            if "ring_bond_type" in mlm_logits and "ring_bond_labels" in mlm_labels:
                pred_ring_bond = mlm_logits["ring_bond_type"]  # (num_big_nodes, num_classes)
                target_ring_bond = mlm_labels["ring_bond_labels"]  # 多标签目标
                
                # 从 batched_graph 获取 ring_bond 概率信息
                rb_probs = batched_graph.rb_probs
                rb_node_ptr = batched_graph.rb_node_ptr
                rb_type_indices = batched_graph.rb_type_indices  # 原始索引
                
                # 使用模型返回的被掩码后的索引来判断哪些被掩码了
                rb_masked_indices = mlm_labels.get("rb_masked_indices", rb_type_indices)
                
                # 获取MASK token索引
                mask_idx = get_mask_token_idx(token_to_idx, "ring_bond_type") if token_to_idx else len(model.big_embeddings["ring_bond_type"].weight)
                
                num_big_nodes = x_dict["big_node"].size(0)
                
                # 损失计算：对所有有效样本（概率=1或只有N/A）
                mask_ring_bond_loss = torch.zeros(num_big_nodes, dtype=torch.bool, device=pred_ring_bond.device)
                
                # 指标统计：只对被掩码的样本（与其他任务保持一致）
                mask_ring_bond_metrics = torch.zeros(num_big_nodes, dtype=torch.bool, device=pred_ring_bond.device)
                
                for i in range(num_big_nodes):
                    start = rb_node_ptr[i].item()
                    end = rb_node_ptr[i+1].item() if i+1 < len(rb_node_ptr) else len(rb_probs)
                    node_rb_probs = rb_probs[start:end]
                    node_rb_types = rb_type_indices[start:end]
                    node_rb_masked = rb_masked_indices[start:end]
                    
                    # 检查该节点是否有概率=1.0的ring_bond
                    has_prob_one = (node_rb_probs >= 1.0).any()
                    
                    # 检查该节点是否有ring_bond被掩码（使用被掩码后的索引）
                    has_masked_rb = (node_rb_masked == mask_idx).any()
                    
                    # 损失计算：只对有有效ring_bond（概率=1）的样本
                    # N/A已被标记为不可预测标签，不参与损失计算
                    if has_prob_one:
                        mask_ring_bond_loss[i] = True
                    
                    # 指标统计：只有被掩码且有有效ring_bond的样本
                    if has_prob_one and has_masked_rb:
                        mask_ring_bond_metrics[i] = True
                
                # 计算损失
                if mask_ring_bond_loss.any():
                    # 多标签分类：使用带权重的BCEWithLogitsLoss
                    if "ring_bond" in mlm_criteria and mlm_criteria["ring_bond"] is not None:
                        # 使用 focal loss 或带权重的损失
                        loss = mlm_criteria["ring_bond"](pred_ring_bond[mask_ring_bond_loss], target_ring_bond[mask_ring_bond_loss].float())
                    else:
                        loss_fct = nn.BCEWithLogitsLoss()
                        loss = loss_fct(pred_ring_bond[mask_ring_bond_loss], target_ring_bond[mask_ring_bond_loss].float())
                    mlm_loss += loss * loss_weights["ring_bond"]
                    task_losses["ring_bond"] += loss.item()
                    task_counts["ring_bond"] += 1
                
                # 收集预测和标签用于计算指标（只对被掩码的样本）
                if mask_ring_bond_metrics.any():
                    preds = (torch.sigmoid(pred_ring_bond[mask_ring_bond_metrics]) >= 0.5).int()
                    task_preds["ring_bond"].append(preds)
                    task_targets["ring_bond"].append(target_ring_bond[mask_ring_bond_metrics].int())
            
            # is_cyclic 二分类损失（预测是否成环）
            # 这是一个特殊的掩码预测任务，只对 ring_bond_type 被掩码的样本计算损失
            # 负样本：ring_bond_type=N/A 被掩码（未成环，is_cyclic=0）
            # 正样本：ring_bond_type!=N/A 被掩码（成环，is_cyclic=1）
            if "is_cyclic" in mlm_logits and "is_cyclic" in mlm_labels and "rb_masked_for_cyclic" in mlm_labels:
                pred_is_cyclic = mlm_logits["is_cyclic"]  # (num_big_nodes, 2)
                target_is_cyclic = mlm_labels["is_cyclic"].squeeze(-1).long()  # (num_big_nodes,)
                rb_masked_for_cyclic = mlm_labels["rb_masked_for_cyclic"]  # ring_bond被掩码的节点
                
                # 只对 ring_bond 被掩码的样本计算损失
                if rb_masked_for_cyclic.any():
                    # 使用交叉熵损失
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(pred_is_cyclic[rb_masked_for_cyclic], target_is_cyclic[rb_masked_for_cyclic])
                    mlm_loss += loss * loss_weights.get("is_cyclic", 1.0)
                    task_losses["is_cyclic"] += loss.item()
                    task_counts["is_cyclic"] += 1
                    # 收集预测和标签用于计算指标
                    task_preds["is_cyclic"].append(pred_is_cyclic[rb_masked_for_cyclic].argmax(dim=1))
                    task_targets["is_cyclic"].append(target_is_cyclic[rb_masked_for_cyclic])
            
            # big_edge特征的MLM损失
            if "big_edge_type" in mlm_logits and "big_edge_attr" in mlm_labels and mlm_labels["big_edge_attr"] is not None:
                pred_big_edge = mlm_logits["big_edge_type"]
                target_big_edge = mlm_labels["big_edge_attr"][:, 0]
                current_big_edge_attr = edge_attr_dict[("big_node", "big_edge", "big_node")][:, 0]
                # 使用正确的MASK token索引判断被掩码的位置
                mask_idx = get_mask_token_idx(token_to_idx, "big_edge_type") if token_to_idx else len(model.big_edge_emb.weight)
                mask_big_edge = (current_big_edge_attr == mask_idx)
                # 同时需要是predictable的边（排除包含?或概率<1.0的边）
                if "big_edge_predictable" in mlm_labels and mlm_labels["big_edge_predictable"] is not None:
                    mask_big_edge = mask_big_edge & mlm_labels["big_edge_predictable"]
                if mask_big_edge.any():
                    loss = mlm_criteria["big_edge"](pred_big_edge[mask_big_edge], target_big_edge[mask_big_edge])
                    mlm_loss += loss * loss_weights["big_edge"]
                    task_losses["big_edge"] += loss.item()
                    task_counts["big_edge"] += 1
                    # 收集预测和标签用于计算指标
                    task_preds["big_edge"].append(pred_big_edge[mask_big_edge].argmax(dim=1))
                    task_targets["big_edge"].append(target_big_edge[mask_big_edge])
            
            # big_edge存在性预测损失（使用正负样本）
            if "big_edge_exists" in mlm_logits and "deleted_big_edge_mask" in mlm_labels:
                pred_big_edge_exists = mlm_logits["big_edge_exists"]
                deleted_mask = mlm_labels["deleted_big_edge_mask"]
                neg_edge_index = mlm_labels.get("negative_big_edge_index")
                
                if deleted_mask is not None and deleted_mask.any():
                    # 正样本标签：被删除的边应该存在（标签为1）
                    num_pos = int(deleted_mask.sum().item())
                    target_pos = torch.ones(num_pos, dtype=torch.long, device=pred_big_edge_exists.device)
                    
                    # 负样本标签：不存在的边不应该存在（标签为0）
                    if neg_edge_index is not None and neg_edge_index.size(1) > 0:
                        num_neg = neg_edge_index.size(1)
                        target_neg = torch.zeros(num_neg, dtype=torch.long, device=pred_big_edge_exists.device)
                        
                        # 拼接正负样本标签
                        target = torch.cat([target_pos, target_neg], dim=0)
                        # 只对被删除的边和负样本计算损失
                        pred = pred_big_edge_exists
                    else:
                        # 没有负样本，只计算正样本
                        target = target_pos
                        pred = pred_big_edge_exists[:num_pos]
                    
                    loss = edge_exists_criterion(pred, target)
                    mlm_loss += loss * loss_weights["big_edge_exists"]
                    task_losses["big_edge_exists"] += loss.item()
                    task_counts["big_edge_exists"] += 1
                    # 收集预测和标签用于计算指标
                    task_preds["big_edge_exists"].append(pred.argmax(dim=1))
                    task_targets["big_edge_exists"].append(target)
            
            # small_edge特征的MLM损失（只计算stereo_type，carbon_position_type不掩码）
            if "stereo_type" in mlm_logits and "small_edge_attr" in mlm_labels and mlm_labels["small_edge_attr"] is not None:
                pred_stereo = mlm_logits["stereo_type"]
                target_stereo = mlm_labels["small_edge_attr"][:, 0]
                current_small_edge_attr = edge_attr_dict[("big_node", "small_edge", "small_node")][:, 0]
                # 使用正确的MASK token索引判断被掩码的位置
                mask_idx = get_mask_token_idx(token_to_idx, "stereo_type") if token_to_idx else len(model.small_edge_stereo_emb.weight)
                mask_stereo = (current_small_edge_attr == mask_idx)
                # 同时需要是predictable的边（排除包含?、是x或概率<1.0的边）
                if "small_edge_predictable" in mlm_labels and mlm_labels["small_edge_predictable"] is not None:
                    mask_stereo = mask_stereo & mlm_labels["small_edge_predictable"]
                if mask_stereo.any():
                    loss = mlm_criteria["small_edge_stereo"](pred_stereo[mask_stereo], target_stereo[mask_stereo])
                    mlm_loss += loss * loss_weights["small_edge_stereo"]
                    task_losses["small_edge_stereo"] += loss.item()
                    task_counts["small_edge_stereo"] += 1
                    # 收集预测和标签用于计算指标
                    task_preds["small_edge_stereo"].append(pred_stereo[mask_stereo].argmax(dim=1))
                    task_targets["small_edge_stereo"].append(target_stereo[mask_stereo])
            
            loss = mlm_loss
        
        # 混合精度训练：使用 scaler 进行梯度缩放和反向传播
        scaler.scale(loss).backward()
        # 梯度裁剪，防止梯度爆炸导致训练不稳定
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    
    # 计算每个任务的平均损失
    avg_task_losses = {}
    for task_name, total in task_losses.items():
        count = max(task_counts[task_name], 1)
        avg_task_losses[task_name] = total / count
    
    # 计算每个任务的分类指标
    task_metrics = compute_task_metrics(task_preds, task_targets, list(task_losses.keys()), vocab_sizes=vocab_sizes, token_to_idx=token_to_idx)
    
    return avg_loss, avg_task_losses, task_metrics


def find_optimal_thresholds(model, loader, label_classes):
    """
    基于训练集（或其他数据集）找到每个类别的最优阈值。
    使用网格搜索在 [0.1, 0.9] 范围内找使 macro F1 最大的阈值。
    返回每个类别的最优阈值。
    
    注意：阈值搜索在CPU上进行，避免GPU内存占用和数据转移开销。
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batched_graph, global_feat, labels, ids, wurcs in loader:
            batched_graph = batched_graph.to(DEVICE)
            global_feat = global_feat.to(DEVICE)

            x_dict = {nt: batched_graph[nt].x for nt in batched_graph.node_types}
            edge_index_dict = {et: batched_graph[et].edge_index for et in batched_graph.edge_types}
            edge_attr_dict = {et: batched_graph[et].edge_attr for et in batched_graph.edge_types}
            edge_prob_dict = {et: batched_graph[et].edge_prob for et in batched_graph.edge_types}
            batch_dict = {nt: batched_graph[nt].batch for nt in batched_graph.node_types}

            logits = model(x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                          batch_dict, global_feat, batched_graph)
            probs = torch.sigmoid(logits)
            
            # 直接转到CPU，避免中间存储在GPU上
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 对每个类别找最优阈值（纯CPU操作）
    # 两阶段搜索：先粗搜索(0.05步长)定位大致范围，再细搜索(0.01步长)精确定位
    optimal_thresholds = []

    for i in range(len(label_classes)):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        # 如果该类别全是正样本或全是负样本，使用默认阈值
        if len(np.unique(y_true)) < 2:
            optimal_thresholds.append(0.5)
            continue

        # ---- 阶段1：粗搜索 (步长0.05) ----
        coarse_thresholds = np.arange(0.1, 0.91, 0.05)
        best_thresh_coarse = 0.5
        best_f1_coarse = 0.0
        for thresh in coarse_thresholds:
            y_pred = (y_score > thresh).astype(int)
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
            y_pred = (y_score > thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds.append(best_thresh)

    return optimal_thresholds


def apply_thresholds(probs, thresholds):
    """使用per-class阈值进行预测"""
    preds = np.zeros_like(probs)
    for i, thresh in enumerate(thresholds):
        preds[:, i] = (probs[:, i] > thresh).astype(int)
    return preds


@torch.no_grad()
def evaluate(model, loader, monitor_criterion, thresholds=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batched_graph, global_feat, labels, ids, wurcs in loader:
        batched_graph = batched_graph.to(DEVICE)
        global_feat = global_feat.to(DEVICE)
        labels = labels.to(DEVICE)

        x_dict = {nt: batched_graph[nt].x for nt in batched_graph.node_types}
        edge_index_dict = {et: batched_graph[et].edge_index for et in batched_graph.edge_types}
        edge_attr_dict = {et: batched_graph[et].edge_attr for et in batched_graph.edge_types}
        edge_prob_dict = {et: batched_graph[et].edge_prob for et in batched_graph.edge_types}
        edge_can_generate_neg_dict = {et: batched_graph[et].can_generate_neg if hasattr(batched_graph[et], 'can_generate_neg') else None 
                                       for et in batched_graph.edge_types}
        edge_src_can_neg_dict = {et: batched_graph[et].src_can_neg if hasattr(batched_graph[et], 'src_can_neg') else None 
                                  for et in batched_graph.edge_types}
        edge_tgt_can_neg_dict = {et: batched_graph[et].tgt_can_neg if hasattr(batched_graph[et], 'tgt_can_neg') else None 
                                  for et in batched_graph.edge_types}
        batch_dict = {nt: batched_graph[nt].batch for nt in batched_graph.node_types}

        logits = model(x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                       batch_dict, global_feat, batched_graph)
        # 用无权重Focal Loss计算val loss，与train monitor loss同一尺度
        loss = monitor_criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits)
        if thresholds is not None:
            # 使用自定义阈值
            probs_np = probs.cpu().numpy()
            preds_np = apply_thresholds(probs_np, thresholds)
            preds = torch.tensor(preds_np, dtype=torch.float32, device=probs.device)
        else:
            preds = (probs > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_micro": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_samples": f1_score(all_labels, all_preds, average="samples", zero_division=0),
        "hamming": hamming_loss(all_labels, all_preds),
        "precision_micro": precision_score(all_labels, all_preds, average="micro", zero_division=0),
        "recall_micro": recall_score(all_labels, all_preds, average="micro", zero_division=0),
    }
    return avg_loss, metrics, all_preds, all_labels, all_probs


@torch.no_grad()
def evaluate_with_details(model, loader, monitor_criterion, thresholds=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_ids = []
    all_wurcs = []
    all_probs = []

    for batched_graph, global_feat, labels, ids, wurcs in loader:
        batched_graph = batched_graph.to(DEVICE)
        global_feat = global_feat.to(DEVICE)
        labels = labels.to(DEVICE)

        x_dict = {nt: batched_graph[nt].x for nt in batched_graph.node_types}
        edge_index_dict = {et: batched_graph[et].edge_index for et in batched_graph.edge_types}
        edge_attr_dict = {et: batched_graph[et].edge_attr for et in batched_graph.edge_types}
        edge_prob_dict = {et: batched_graph[et].edge_prob for et in batched_graph.edge_types}
        edge_can_generate_neg_dict = {et: batched_graph[et].can_generate_neg if hasattr(batched_graph[et], 'can_generate_neg') else None 
                                       for et in batched_graph.edge_types}
        edge_src_can_neg_dict = {et: batched_graph[et].src_can_neg if hasattr(batched_graph[et], 'src_can_neg') else None 
                                  for et in batched_graph.edge_types}
        edge_tgt_can_neg_dict = {et: batched_graph[et].tgt_can_neg if hasattr(batched_graph[et], 'tgt_can_neg') else None 
                                  for et in batched_graph.edge_types}
        batch_dict = {nt: batched_graph[nt].batch for nt in batched_graph.node_types}

        logits = model(x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                       batch_dict, global_feat, batched_graph)
        # 用无权重Focal Loss计算loss，与train/val monitor loss同一尺度
        loss = monitor_criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits)
        if thresholds is not None:
            # 使用自定义阈值
            probs_np = probs.cpu().numpy()
            preds_np = apply_thresholds(probs_np, thresholds)
            preds = torch.tensor(preds_np, dtype=torch.float32, device=probs.device)
        else:
            preds = (probs > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_ids.extend(ids)
        all_wurcs.extend(wurcs)

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    error_details = []
    for i in range(len(all_ids)):
        if not np.array_equal(all_preds[i], all_labels[i]):
            true_labels_str = "_".join(
                [LABEL_CLASSES[j] for j in range(len(LABEL_CLASSES)) if all_labels[i][j] == 1]
            )
            pred_labels_str = "_".join(
                [LABEL_CLASSES[j] for j in range(len(LABEL_CLASSES)) if all_preds[i][j] == 1]
            )
            if not pred_labels_str:
                pred_labels_str = "None"
            error_details.append({
                "glytoucan_id": all_ids[i],
                "wurcs": all_wurcs[i],
                "true_label": true_labels_str,
                "pred_label": pred_labels_str,
                "probabilities": {
                    LABEL_CLASSES[j]: float(all_probs[i][j]) for j in range(len(LABEL_CLASSES))
                }
            })

    # ---- 计算 AUC ----
    per_label_auc = {}
    valid_auc_labels = []
    for i, cls_name in enumerate(LABEL_CLASSES):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        # 只有当正负样本都存在时才能计算AUC
        if len(np.unique(y_true)) > 1:
            try:
                per_label_auc[cls_name] = float(roc_auc_score(y_true, y_score))
                valid_auc_labels.append(i)
            except ValueError:
                per_label_auc[cls_name] = None
        else:
            per_label_auc[cls_name] = None

    # micro AUC: 将所有标签展平后计算
    try:
        auc_micro = float(roc_auc_score(all_labels, all_probs, average="micro"))
    except ValueError:
        auc_micro = None

    # macro AUC: 各标签AUC取平均（仅对可计算AUC的标签）
    if valid_auc_labels:
        auc_macro = float(np.mean([per_label_auc[LABEL_CLASSES[i]] for i in valid_auc_labels]))
    else:
        auc_macro = None

    # ---- 计算 Per-label Precision, Recall, F1 ----
    per_label_precision = {}
    per_label_recall = {}
    per_label_f1 = {}
    for i, cls_name in enumerate(LABEL_CLASSES):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        per_label_precision[cls_name] = float(precision_score(y_true, y_pred, zero_division=0))
        per_label_recall[cls_name] = float(recall_score(y_true, y_pred, zero_division=0))
        per_label_f1[cls_name] = float(f1_score(y_true, y_pred, zero_division=0))

    # ---- 计算 Macro Precision, Recall ----
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # ---- 计算样本级指标 ----
    # 样本完全匹配率（所有标签都预测正确）
    exact_match = (all_preds == all_labels).all(axis=1).mean()
    
    # 样本级精度（预测正确的标签占预测标签的比例）
    all_preds_int = all_preds.astype(np.int32)
    all_labels_int = all_labels.astype(np.int32)
    pred_sum = all_preds_int.sum(axis=1)
    correct_pred = (all_preds_int & all_labels_int).sum(axis=1)
    correct_pred = correct_pred.astype(float)
    sample_precision = (correct_pred / np.maximum(pred_sum, 1)).mean()
    
    # 样本级召回率（预测正确的标签占真实标签的比例）
    true_sum = all_labels_int.sum(axis=1)
    sample_recall = (correct_pred / np.maximum(true_sum, 1)).mean()

    # ---- 阈值分析：统计不同阈值下的正样本预测情况 ----
    threshold_stats = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_at_thresh = (all_probs > thresh).astype(int)
        threshold_stats[f"f1_macro_thresh_{thresh}"] = float(f1_score(all_labels, preds_at_thresh, average="macro", zero_division=0))
        threshold_stats[f"precision_macro_thresh_{thresh}"] = float(precision_score(all_labels, preds_at_thresh, average="macro", zero_division=0))
        threshold_stats[f"recall_macro_thresh_{thresh}"] = float(recall_score(all_labels, preds_at_thresh, average="macro", zero_division=0))

    # ---- 类别不平衡分析 ----
    # 每个类别的正样本数和预测正样本数
    class_stats = {}
    for i, cls_name in enumerate(LABEL_CLASSES):
        n_true_pos = int(all_labels[:, i].sum())
        n_pred_pos = int(all_preds[:, i].sum())
        class_stats[cls_name] = {
            "true_positive_count": n_true_pos,
            "predicted_positive_count": n_pred_pos,
            "support": len(all_labels),
            "prevalence": float(n_true_pos / len(all_labels)),
        }

    # ---- 混淆矩阵统计 ----
    # 计算总体的 TN, FP, FN, TP
    tn_total = ((all_labels == 0) & (all_preds == 0)).sum()
    fp_total = ((all_labels == 0) & (all_preds == 1)).sum()
    fn_total = ((all_labels == 1) & (all_preds == 0)).sum()
    tp_total = ((all_labels == 1) & (all_preds == 1)).sum()

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "exact_match": float(exact_match),
        "sample_precision": float(sample_precision),
        "sample_recall": float(sample_recall),
        "f1_micro": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_samples": f1_score(all_labels, all_preds, average="samples", zero_division=0),
        "precision_micro": precision_score(all_labels, all_preds, average="micro", zero_division=0),
        "recall_micro": recall_score(all_labels, all_preds, average="micro", zero_division=0),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "hamming": hamming_loss(all_labels, all_preds),
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
        "per_label_auc": per_label_auc,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "per_label_f1": per_label_f1,
        "threshold_stats": threshold_stats,
        "class_stats": class_stats,
        "confusion_matrix_total": {
            "tn": int(tn_total),
            "fp": int(fp_total),
            "fn": int(fn_total),
            "tp": int(tp_total),
            "total": int(tn_total + fp_total + fn_total + tp_total),
        },
    }
    return avg_loss, metrics, error_details, all_preds, all_labels


def plot_confusion_matrices(all_labels, all_preds, label_classes, save_dir, fig_width_per_label=2.0, fig_height_per_label=2.0):
    """
    为多标签分类任务生成混淆矩阵图。
    
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
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        ax_single.set_title(
            f"Confusion Matrix: {cls_name}\n"
            f"(n={int(all_labels[:, i].sum())}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f})",
            fontsize=11, pad=10
        )
        ax_single.set_ylabel("True Label", fontsize=10)
        ax_single.set_xlabel("Predicted Label", fontsize=10)

        plt.tight_layout()

        # 文件名中替换不合法字符
        safe_name = cls_name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_") \
            .replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
        single_fig_path = os.path.join(individual_dir, f"{safe_name}.png")
        fig_single.savefig(single_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig_single)

    print(f"Individual label confusion matrices saved to: {individual_dir}/")


def main():
    log_path = os.path.join(ERROR_ANALYSIS_DIR, "training_log.txt")
    sys.stdout = Logger(log_path)
    print("=" * 60)
    print("Loading vocabulary...")
    vocab_sizes, token_to_idx = load_vocab(VOCAB_PATH)
    print(f"Vocabulary categories: {list(vocab_sizes.keys())}")
    for k, v in vocab_sizes.items():
        print(f"  {k}: {v} tokens")

    print("\n" + "=" * 60)
    print("Loading data...")
    raw_data = load_all_data(OUTPUT_ALL_PATH)
    print(f"Total samples in raw data: {len(raw_data)}")

    # ========== 使用全部数据训练 ==========
    MAX_SAMPLES = None # 设置为None则使用全部数据，设置为数字则限制样本数量
    if MAX_SAMPLES is not None and MAX_SAMPLES < len(raw_data):
        raw_data = raw_data[:MAX_SAMPLES]
        print(f"[TEST MODE] Limited to {len(raw_data)} samples")

    print("\n" + "=" * 60)
    print("Building dataset...")
    full_dataset = GlycanDataset(raw_data, token_to_idx)

    total = len(full_dataset)
    indices = list(range(total))

    # 在10%数据中划分：80%训练，20%验证
    train_size = int(total * 0.8)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"Train set: {len(train_indices)} samples ({len(train_indices) / total * 100:.1f}%)")
    print(f"Val set: {len(val_indices)} samples ({len(val_indices) / total * 100:.1f}%)")


    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"\nUsing full data: train={len(train_subset)}, val={len(val_subset)}")

    # 计算每个任务的类别平衡权重（用于Focal Loss）
    # 使用全部训练集统计类别分布（None表示全部）
    num_samples_for_stats = None
    task_criteria = compute_task_class_weights(train_subset, token_to_idx, vocab_sizes, num_samples_for_stats=num_samples_for_stats)

    graph_sizes_full = {}
    for i in range(len(full_dataset)):
        g, _, _, _ = full_dataset[i]
        node_count = g["big_node"].x.size(0) + g["small_node"].x.size(0)
        # 边数：概率边也算1条边，显存不关心概率
        edge_count = 0
        for etype in g.edge_types:
            edge_count += g[etype].edge_index.size(1)
        # 节点数+边数之和，协同控制显存
        graph_sizes_full[i] = node_count + edge_count

    def make_local_sizes(indices_list):
        return {i: graph_sizes_full[orig] for i, orig in enumerate(indices_list)}

    train_local_sizes = make_local_sizes(train_indices)
    val_local_sizes = make_local_sizes(val_indices)

    TARGET_SIZE_PER_BATCH = 588888 # 增大batch size提高GPU利用率
    num_workers = 0  # Windows下多进程会卡住，设为0使用单进程
    pin_memory = False  # Windows下pin_memory可能导致问题，设为False

    train_sampler = DynamicBatchSampler(
        list(range(len(train_indices))), train_local_sizes,
        target_size_per_batch=TARGET_SIZE_PER_BATCH, shuffle=True
    )
    val_sampler = DynamicBatchSampler(
        list(range(len(val_indices))), val_local_sizes,
        target_size_per_batch=TARGET_SIZE_PER_BATCH, shuffle=False
    )

    print(f"\nDynamic batching: target ~{TARGET_SIZE_PER_BATCH} (nodes+edges)/batch")
    print(f"  Train batches: {len(train_sampler)} (avg {len(train_indices) / max(len(train_sampler), 1):.0f} graphs/batch)")
    print(f"  Val batches:   {len(val_sampler)}")

    # 启用CUDA优化
    torch.backends.cudnn.benchmark = True  # 优化卷积运算

    train_loader = DataLoader(
        train_subset, batch_sampler=train_sampler,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_subset, batch_sampler=val_sampler,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )

    print("\n" + "=" * 60)
    print("Building model...")
    freq_stats_path = os.path.join(os.path.dirname(VOCAB_PATH), "vocab_freq_stats.txt")
    model = HeteroGNN(
        vocab_sizes=vocab_sizes,
        hidden_dim=256,
        num_layers=4,
        dropout=0.2,  # 降低正则化，解决欠拟合问题
        num_heads=4,
        mlm_probability=0.30,  # 提高掩码率，测试数据上限
        freq_stats_path=freq_stats_path,
        token_to_idx=token_to_idx,  # 传递token_to_idx用于获取MASK token索引
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # 降低正则化，提高学习率
    
    # 混合精度训练：初始化 GradScaler
    scaler = GradScaler()

    early_stop_patience = 20
    ema_window = 5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5,
                                                           factor=0.5, min_lr=1e-6)
    # 使用EMA平滑后的验证损失监控早停和学习率递减
    early_stopper = EarlyStopping(patience=early_stop_patience, min_delta=1e-4, mode="min",
                                   ema_window=ema_window, use_ema=True)

    MAX_EPOCHS = 200

    best_val_loss = float('inf')  # 保存MLM损失最低的模型
    best_epoch = 0
    best_model_state = None

    print("\n" + "=" * 60)
    print(f"Training config (MLM Pre-training):")
    print(f"  Target (nodes+edges)/batch: {TARGET_SIZE_PER_BATCH}")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: {pin_memory}")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Loss: MLM Cross Entropy Loss")
    print(f"  Early stop patience: {early_stop_patience} (monitor: ema_val_loss)")
    print(f"  Optimizer: AdamW, lr=1e-3, weight_decay=1e-4")
    print(f"  Scheduler: ReduceLROnPlateau, patience=5, factor=0.5, min_lr=1e-6")
    print(f"  Gradient clip: 1.0")
    print(f"  MLM probability: 0.30")
    print("=" * 60)

    # 定义验证函数
    def evaluate_mlm(model, loader, task_criteria=None, vocab_sizes=None, token_to_idx=None):
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # 为每个任务创建带有类别平衡权重的 Focal Loss（与训练一致）
        if task_criteria is None:
            task_criteria = {}
        mlm_criteria = {}
        for task_name in ["carbon_count", "anomeric_config", "ring_bond", "substituent", "big_edge", "small_edge_stereo"]:
            if task_name in task_criteria and task_criteria[task_name] is not None:
                mlm_criteria[task_name] = MLMFocalLoss(gamma=3.0, alpha=task_criteria[task_name])
            else:
                mlm_criteria[task_name] = MLMFocalLoss(gamma=3.0)
        
        # 边存在性预测使用二元交叉熵损失
        edge_exists_criterion = nn.CrossEntropyLoss()
        
        # 记录每个任务的损失
        task_losses = {
            # "anomeric_position": 0.0,  # 已删除，不再作为预测任务
            "carbon_count": 0.0,        # 新增：碳数预测
            "anomeric_config": 0.0,
            "ring_bond": 0.0,           # ring_bond多标签预测，只对概率为1的计算损失
            "substituent": 0.0,         # 取代基预测任务，O不参与loss计算
            "big_edge": 0.0,
            "small_edge_stereo": 0.0,
            "big_edge_exists": 0.0,     # 新增：大环边存在性预测
            "is_cyclic": 0.0,           # 新增：是否成环二分类任务
        }
        task_counts = {k: 0 for k in task_losses.keys()}
        
        # 收集每个任务的预测和标签，用于计算分类指标
        task_preds = {k: [] for k in task_losses.keys()}
        task_targets = {k: [] for k in task_losses.keys()}
        
        with torch.no_grad():
            for batched_graph, global_feat, labels, ids, wurcs in loader:
                batched_graph = batched_graph.to(DEVICE)
                global_feat = global_feat.to(DEVICE)

                x_dict = {nt: batched_graph[nt].x for nt in batched_graph.node_types}
                edge_index_dict = {et: batched_graph[et].edge_index for et in batched_graph.edge_types}
                edge_attr_dict = {et: batched_graph[et].edge_attr for et in batched_graph.edge_types}
                edge_prob_dict = {et: batched_graph[et].edge_prob for et in batched_graph.edge_types}
                batch_dict = {nt: batched_graph[nt].batch for nt in batched_graph.node_types}

                features, mlm_logits, mlm_labels = model(x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                                                       batch_dict, global_feat, batched_graph, return_mlm_logits=True)

                mlm_loss = 0.0
                
                # 任务损失权重配置 - 与训练一致
                loss_weights = {
                    # "anomeric_position": 1.0,  # 已删除，不再作为预测任务
                    "carbon_count": 1.0,
                    "anomeric_config": 1.0,
                    "ring_bond": 1.0,        # ring_bond多标签预测权重，只对概率为1的计算损失
                    "substituent": 1.0,         # 取代基预测任务权重，O不参与loss计算
                    "big_edge": 1.0,
                    "small_edge_stereo": 1.0,
                    "big_edge_exists": 1.0,
                    "is_cyclic": 1.0,        # 是否成环二分类任务权重
                }
                

                for i, (feat_name, task_name) in enumerate([("carbon_count_type", "carbon_count"), 
                                                            ("anomeric_config_type", "anomeric_config")]):
                    col = [0, 2][i]  # 对应big_x的第0列（碳数）和第2列（anomeric_config）
                    pred = mlm_logits[feat_name]
                    target = mlm_labels["big_x"][:, col]
                    # 使用正确的MASK token索引判断被掩码的位置
                    mask_idx = get_mask_token_idx(token_to_idx, feat_name) if token_to_idx else vocab_sizes.get(feat_name, 10)
                    mask = (x_dict["big_node"][:, col] == mask_idx)
                    # 获取predictable标记
                    if hasattr(batched_graph["big_node"], 'predictable'):
                        predictable = batched_graph["big_node"].predictable.squeeze(-1).bool()
                        mask = mask & predictable
                    if mask.any():
                        loss = mlm_criteria[task_name](pred[mask], target[mask])
                        mlm_loss += loss * loss_weights[task_name]
                        task_losses[task_name] += loss.item()
                        task_counts[task_name] += 1
                        # 收集预测和标签用于计算指标
                        task_preds[task_name].append(pred[mask].argmax(dim=1))
                        task_targets[task_name].append(target[mask])

                # small_node substituent 特征的MLM损失（O不参与loss计算）
                if "substituent" in mlm_logits:
                    pred_substituent = mlm_logits["substituent"]
                    target_substituent = mlm_labels["small_x"][:, 0]
                    current_small_x = x_dict["small_node"][:, 0]
                    # 使用正确的MASK token索引判断被掩码的位置
                    mask_idx = get_mask_token_idx(token_to_idx, "substituent") if token_to_idx else vocab_sizes.get("substituent", 33)
                    mask_substituent = (current_small_x == mask_idx)
                    # 排除O（无需预测项），动态获取O的索引
                    o_idx = token_to_idx["substituent"].get("O", 13) if token_to_idx else 13
                    not_o_mask = (target_substituent != o_idx)
                    mask_substituent = mask_substituent & not_o_mask
                    if mask_substituent.any():
                        loss = mlm_criteria["substituent"](pred_substituent[mask_substituent], target_substituent[mask_substituent])
                        mlm_loss += loss * loss_weights["substituent"]
                        task_losses["substituent"] += loss.item()
                        task_counts["substituent"] += 1
                        # 收集预测和标签用于计算指标（排除O）
                        task_preds["substituent"].append(pred_substituent[mask_substituent].argmax(dim=1))
                        task_targets["substituent"].append(target_substituent[mask_substituent])
                
                # ring_bond 特征的多标签MLM损失（只对概率为1的环键计算损失）
                if "ring_bond_type" in mlm_logits and "ring_bond_labels" in mlm_labels:
                    pred_ring_bond = mlm_logits["ring_bond_type"]  # (num_big_nodes, num_classes)
                    target_ring_bond = mlm_labels["ring_bond_labels"]  # 多标签目标
                    
                    # 从 batched_graph 获取 ring_bond 概率信息
                    rb_probs = batched_graph.rb_probs
                    rb_node_ptr = batched_graph.rb_node_ptr
                    
                    # 使用模型返回的原始索引（未被掩码的）来判断是否只有N/A
                    rb_original_indices = mlm_labels.get("rb_indices", batched_graph.rb_type_indices)
                    
                    # 使用模型返回的被掩码后的索引来判断哪些被掩码了
                    rb_masked_indices = mlm_labels.get("rb_masked_indices", rb_original_indices)
                    
                    # 获取MASK token索引
                    mask_idx = get_mask_token_idx(token_to_idx, "ring_bond_type") if token_to_idx else vocab_sizes.get("ring_bond_type", 12)
                    
                    # 获取N/A的索引（索引0）
                    na_idx = token_to_idx["ring_bond_type"].get("N/A", 0) if token_to_idx else 0
                    
                    num_big_nodes = x_dict["big_node"].size(0)
                    
                    # 损失计算：对所有有效样本（概率=1或只有N/A）
                    mask_ring_bond_loss = torch.zeros(num_big_nodes, dtype=torch.bool, device=pred_ring_bond.device)
                    
                    # 指标统计：只对被掩码的样本（与其他任务保持一致）
                    mask_ring_bond_metrics = torch.zeros(num_big_nodes, dtype=torch.bool, device=pred_ring_bond.device)
                    
                    for i in range(num_big_nodes):
                        start = rb_node_ptr[i].item()
                        end = rb_node_ptr[i+1].item() if i+1 < len(rb_node_ptr) else len(rb_probs)
                        node_rb_probs = rb_probs[start:end]
                        node_rb_types = rb_original_indices[start:end]  # 使用原始索引判断N/A
                        node_rb_masked = rb_masked_indices[start:end]
                        
                        # 检查该节点是否有概率=1.0的ring_bond
                        has_prob_one = (node_rb_probs >= 1.0).any()
                        
                        # 检查该节点是否有ring_bond被掩码（使用被掩码后的索引）
                        has_masked_rb = (node_rb_masked == mask_idx).any()
                        
                        # 损失计算：只对有有效ring_bond（概率=1）的样本
                        # N/A已被标记为不可预测标签，不参与损失计算
                        if has_prob_one:
                            mask_ring_bond_loss[i] = True
                        
                        # 指标统计：只有被掩码且有有效ring_bond的样本
                        if has_prob_one and has_masked_rb:
                            mask_ring_bond_metrics[i] = True
                    
                    # 计算损失
                    if mask_ring_bond_loss.any():
                        # 多标签分类：使用BCEWithLogitsLoss
                        loss_fct = nn.BCEWithLogitsLoss()
                        loss = loss_fct(pred_ring_bond[mask_ring_bond_loss], target_ring_bond[mask_ring_bond_loss].float())
                        mlm_loss += loss * loss_weights["ring_bond"]
                        task_losses["ring_bond"] += loss.item()
                        task_counts["ring_bond"] += 1
                    
                    # 收集预测和标签用于计算指标（只对被掩码的样本）
                    if mask_ring_bond_metrics.any():
                        preds = (torch.sigmoid(pred_ring_bond[mask_ring_bond_metrics]) >= 0.5).int()
                        task_preds["ring_bond"].append(preds)
                        task_targets["ring_bond"].append(target_ring_bond[mask_ring_bond_metrics].int())
                
                # is_cyclic 二分类损失（预测是否成环）
                # 这是一个特殊的掩码预测任务，只对 ring_bond_type 被掩码的样本计算损失
                if "is_cyclic" in mlm_logits and "is_cyclic" in mlm_labels and "rb_masked_for_cyclic" in mlm_labels:
                    pred_is_cyclic = mlm_logits["is_cyclic"]  # (num_big_nodes, 2)
                    target_is_cyclic = mlm_labels["is_cyclic"].squeeze(-1).long()  # (num_big_nodes,)
                    rb_masked_for_cyclic = mlm_labels["rb_masked_for_cyclic"]  # ring_bond被掩码的节点
                    
                    if rb_masked_for_cyclic.any():
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(pred_is_cyclic[rb_masked_for_cyclic], target_is_cyclic[rb_masked_for_cyclic])
                        mlm_loss += loss * loss_weights.get("is_cyclic", 1.0)
                        task_losses["is_cyclic"] += loss.item()
                        task_counts["is_cyclic"] += 1
                        # 收集预测和标签用于计算指标
                        task_preds["is_cyclic"].append(pred_is_cyclic[rb_masked_for_cyclic].argmax(dim=1))
                        task_targets["is_cyclic"].append(target_is_cyclic[rb_masked_for_cyclic])
                
                # big_edge特征的MLM损失
                if "big_edge_type" in mlm_logits and "big_edge_attr" in mlm_labels and mlm_labels["big_edge_attr"] is not None:
                    pred_big_edge = mlm_logits["big_edge_type"]
                    target_big_edge = mlm_labels["big_edge_attr"][:, 0]
                    current_big_edge_attr = edge_attr_dict[("big_node", "big_edge", "big_node")][:, 0]
                    # 使用正确的MASK token索引判断被掩码的位置
                    mask_idx = get_mask_token_idx(token_to_idx, "big_edge_type") if token_to_idx else vocab_sizes.get("big_edge_type", 35)
                    mask_big_edge = (current_big_edge_attr == mask_idx)
                    # 同时需要是predictable的边（排除包含?或概率<1.0的边）
                    if "big_edge_predictable" in mlm_labels and mlm_labels["big_edge_predictable"] is not None:
                        mask_big_edge = mask_big_edge & mlm_labels["big_edge_predictable"]
                    if mask_big_edge.any():
                        loss = mlm_criteria["big_edge"](pred_big_edge[mask_big_edge], target_big_edge[mask_big_edge])
                        mlm_loss += loss * loss_weights["big_edge"]
                        task_losses["big_edge"] += loss.item()
                        task_counts["big_edge"] += 1
                        # 收集预测和标签用于计算指标
                        task_preds["big_edge"].append(pred_big_edge[mask_big_edge].argmax(dim=1))
                        task_targets["big_edge"].append(target_big_edge[mask_big_edge])
                
                # big_edge存在性预测损失（使用正负样本）
                if "big_edge_exists" in mlm_logits and "deleted_big_edge_mask" in mlm_labels:
                    pred_big_edge_exists = mlm_logits["big_edge_exists"]
                    deleted_mask = mlm_labels["deleted_big_edge_mask"]
                    neg_edge_index = mlm_labels.get("negative_big_edge_index")
                    
                    if deleted_mask is not None and deleted_mask.any():
                        # 正样本标签：被删除的边应该存在（标签为1）
                        num_pos = int(deleted_mask.sum().item())
                        target_pos = torch.ones(num_pos, dtype=torch.long, device=pred_big_edge_exists.device)
                        
                        # 负样本标签：不存在的边不应该存在（标签为0）
                        if neg_edge_index is not None and neg_edge_index.size(1) > 0:
                            num_neg = neg_edge_index.size(1)
                            target_neg = torch.zeros(num_neg, dtype=torch.long, device=pred_big_edge_exists.device)
                            
                            # 拼接正负样本标签
                            target = torch.cat([target_pos, target_neg], dim=0)
                            # 只对被删除的边和负样本计算损失
                            pred = pred_big_edge_exists
                        else:
                            # 没有负样本，只计算正样本
                            target = target_pos
                            pred = pred_big_edge_exists[:num_pos]
                        
                        loss = edge_exists_criterion(pred, target)
                        mlm_loss += loss * loss_weights["big_edge_exists"]
                        task_losses["big_edge_exists"] += loss.item()
                        task_counts["big_edge_exists"] += 1
                        # 收集预测和标签用于计算指标
                        task_preds["big_edge_exists"].append(pred.argmax(dim=1))
                        task_targets["big_edge_exists"].append(target)
                
                # small_edge特征的MLM损失（只计算stereo_type，carbon_position_type不掩码）
                if "stereo_type" in mlm_logits and "small_edge_attr" in mlm_labels and mlm_labels["small_edge_attr"] is not None:
                    pred_stereo = mlm_logits["stereo_type"]
                    target_stereo = mlm_labels["small_edge_attr"][:, 0]
                    current_small_edge_attr = edge_attr_dict[("big_node", "small_edge", "small_node")][:, 0]
                    # 使用正确的MASK token索引判断被掩码的位置
                    mask_idx = get_mask_token_idx(token_to_idx, "stereo_type") if token_to_idx else vocab_sizes.get("stereo_type", 15)
                    mask_stereo = (current_small_edge_attr == mask_idx)
                    # 同时需要是predictable的边（排除包含?、是x或概率<1.0的边）
                    if "small_edge_predictable" in mlm_labels and mlm_labels["small_edge_predictable"] is not None:
                        mask_stereo = mask_stereo & mlm_labels["small_edge_predictable"]
                    if mask_stereo.any():
                        loss = mlm_criteria["small_edge_stereo"](pred_stereo[mask_stereo], target_stereo[mask_stereo])
                        mlm_loss += loss * loss_weights["small_edge_stereo"]
                        task_losses["small_edge_stereo"] += loss.item()
                        task_counts["small_edge_stereo"] += 1
                        # 收集预测和标签用于计算指标
                        task_preds["small_edge_stereo"].append(pred_stereo[mask_stereo].argmax(dim=1))
                        task_targets["small_edge_stereo"].append(target_stereo[mask_stereo])

                total_loss += mlm_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 计算每个任务的平均损失
        avg_task_losses = {}
        for task_name, total in task_losses.items():
            count = max(task_counts[task_name], 1)
            avg_task_losses[task_name] = total / count
        
        # 计算每个任务的分类指标
        task_metrics = compute_task_metrics(task_preds, task_targets, list(task_losses.keys()), vocab_sizes=vocab_sizes, token_to_idx=token_to_idx)
        
        return avg_loss, avg_task_losses, task_metrics

    # 创建训练指标日志文件
    metrics_log_path = os.path.join(DATA_DIR, "training_metrics_log.txt")
    metrics_log_file = open(metrics_log_path, "w", encoding="utf-8")
    # 写入对齐的表头
    metrics_log_file.write(f"{'Epoch':^6} | {'Split':^6} | {'Task':^20} | {'Loss':^8} | {'Accuracy':^8} | {'Macro_F1':^8} | {'Recall':^8} | {'Precision':^10} | {'Count':^8}\n")
    metrics_log_file.write("-" * 120 + "\n")
    metrics_log_file.flush()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Starting Epoch {epoch}/{MAX_EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_task_losses, train_task_metrics = train_one_epoch(model, train_loader, optimizer, scaler,
                                                        max_grad_norm=1.0, task_criteria=task_criteria, vocab_sizes=vocab_sizes, token_to_idx=token_to_idx, epoch=epoch)

        # 评估验证集
        val_loss, val_task_losses, val_task_metrics = evaluate_mlm(model, val_loader, task_criteria=task_criteria, vocab_sizes=vocab_sizes, token_to_idx=token_to_idx)

        # 更新早停和学习率（均使用EMA5平滑后的验证损失）
        stopped = early_stopper(val_loss)
        # 使用EMA值进行学习率调度，与早停保持一致
        ema_val_loss = early_stopper.ema_value if early_stopper.use_ema else val_loss
        scheduler.step(ema_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # 保存最优模型（基于验证集MLM损失），并实时保存到磁盘
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            # 实时保存最佳模型到磁盘，防止训练中断丢失
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_mlm.pt")
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": best_model_state,
                "val_mlm_loss": best_val_loss,
                "vocab_sizes": vocab_sizes,
                "hidden_dim": 256,
                "num_layers": 4,
                "num_heads": 4,
                "dropout": 0.2,
                "mlm_probability": 0.30,
            }, checkpoint_path)
            print(f"  [Saved best model to {checkpoint_path}]")

        # 获取EMA5值用于显示
        ema_display = early_stopper.ema_value if early_stopper.use_ema else val_loss
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"Train MLM Loss: {train_loss:.4f} | "
              f"Val MLM Loss: {val_loss:.4f} | "
              f"Val Loss EMA5: {ema_display:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"ES: {early_stopper.counter}/{early_stop_patience}")
        
        # 打印训练集各任务的损失
        print(f"  Train Tasks: "
              f"carbon_cnt={train_task_losses['carbon_count']:.4f} | "
              f"anomeric_cfg={train_task_losses['anomeric_config']:.4f} | "
              f"substituent={train_task_losses['substituent']:.4f} | "
              f"ring_bond={train_task_losses['ring_bond']:.4f} | "
              f"is_cyclic={train_task_losses['is_cyclic']:.4f} | "
              f"big_edge={train_task_losses['big_edge']:.4f} | "
              f"stereo={train_task_losses['small_edge_stereo']:.4f} | "
              f"big_edge_exist={train_task_losses['big_edge_exists']:.4f}")
        
        # 打印验证集各任务的损失
        print(f"  Val Tasks:   "
              f"carbon_cnt={val_task_losses['carbon_count']:.4f} | "
              f"anomeric_cfg={val_task_losses['anomeric_config']:.4f} | "
              f"substituent={val_task_losses['substituent']:.4f} | "
              f"ring_bond={val_task_losses['ring_bond']:.4f} | "
              f"is_cyclic={val_task_losses['is_cyclic']:.4f} | "
              f"big_edge={val_task_losses['big_edge']:.4f} | "
              f"stereo={val_task_losses['small_edge_stereo']:.4f} | "
              f"big_edge_exist={val_task_losses['big_edge_exists']:.4f}")

        # 写入指标到 txt 日志文件
        task_names = ["carbon_count", "anomeric_config", "substituent", "ring_bond", 
                      "is_cyclic", "big_edge", "small_edge_stereo", "big_edge_exists"]
        for task_name in task_names:
            # 训练集指标
            train_m = train_task_metrics.get(task_name, {})
            metrics_log_file.write(
                f"{epoch:^6d} | {'Train':^6} | {task_name:<20} | "
                f"{train_task_losses.get(task_name, 0):>8.4f} | "
                f"{train_m.get('accuracy', 0):>8.4f} | "
                f"{train_m.get('macro_f1', 0):>8.4f} | "
                f"{train_m.get('recall', 0):>8.4f} | "
                f"{train_m.get('precision', 0):>10.4f} | "
                f"{train_m.get('count', 0):>8d}\n"
            )
            
            # 验证集指标
            val_m = val_task_metrics.get(task_name, {})
            metrics_log_file.write(
                f"{epoch:^6d} | {'Val':^6} | {task_name:<20} | "
                f"{val_task_losses.get(task_name, 0):>8.4f} | "
                f"{val_m.get('accuracy', 0):>8.4f} | "
                f"{val_m.get('macro_f1', 0):>8.4f} | "
                f"{val_m.get('recall', 0):>8.4f} | "
                f"{val_m.get('precision', 0):>10.4f} | "
                f"{val_m.get('count', 0):>8d}\n"
            )
            # 写入验证集每个标签的 F1 分数和样本数量（使用标签名称）
            per_class_f1 = val_m.get('per_class_f1', {})
            per_class_count = val_m.get('per_class_count', {})
            unpredictable_labels = val_m.get('unpredictable_labels', {})
            label_names = val_m.get('label_names', {})
            if per_class_f1:
                metrics_log_file.write(f"         | {'Val':^6} | {task_name:<20} | Per-Class F1: ")
                f1_strs = []
                for label_idx in sorted(per_class_f1.keys()):
                    # 使用标签名称
                    label_name = label_names.get(label_idx, f"Label_{label_idx}")
                    count = per_class_count.get(label_idx, 0)
                    f1 = per_class_f1[label_idx]
                    
                    # 判断标签状态
                    if label_idx in unpredictable_labels:
                        # 不可预测的标签
                        f1_strs.append(f"{label_name}=无需计算")
                    elif f1 is None or count == 0:
                        # 该标签在验证集中不存在样本
                        f1_strs.append(f"{label_name}=MISSING(n=0)")
                    else:
                        f1_strs.append(f"{label_name}={f1:.4f}(n={count})")
                metrics_log_file.write(" | ".join(f1_strs) + "\n")
        metrics_log_file.flush()

        # 每轮训练之间加空行（最后一轮不加）
        if epoch < MAX_EPOCHS:
            metrics_log_file.write("\n")

        if stopped:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # 关闭指标日志文件
    metrics_log_file.close()
    print(f"\nTraining metrics saved to: {metrics_log_path}")

    print("\n" + "=" * 60)
    print(f"Loading best model from epoch {best_epoch} (val_mlm_loss={best_val_loss:.4f})")
    model.load_state_dict(best_model_state)

    # 最佳模型已在训练过程中实时保存
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_mlm.pt")
    print(f"Best model already saved to: {checkpoint_path}")

    # 验证特征提取输出维度
    print("\n" + "=" * 60)
    print("Verifying feature extraction output dimension...")
    model.eval()
    with torch.no_grad():
        for batched_graph, global_feat, labels, ids, wurcs in val_loader:
            batched_graph = batched_graph.to(DEVICE)
            global_feat = global_feat.to(DEVICE)
            x_dict = {nt: batched_graph[nt].x for nt in batched_graph.node_types}
            edge_index_dict = {et: batched_graph[et].edge_index for et in batched_graph.edge_types}
            edge_attr_dict = {et: batched_graph[et].edge_attr for et in batched_graph.edge_types}
            edge_prob_dict = {et: batched_graph[et].edge_prob for et in batched_graph.edge_types}
            batch_dict = {nt: batched_graph[nt].batch for nt in batched_graph.node_types}
            
            features = model(x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                           batch_dict, global_feat, batched_graph)
            print(f"Feature output shape: {features.shape}")
            print(f"Feature dimension: {features.size(1)}")
            assert features.size(1) == 1027, f"Expected 1027 features, got {features.size(1)}"
            print("✓ Feature extraction output dimension verified: 1027")
            break

    print("\n" + "=" * 60)
    print("MLM Pre-training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best validation MLM loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {checkpoint_path}")
    print(f"  Feature output dimension: 1027")
    print("=" * 60)
    # 恢复标准输出并关闭日志文件
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()