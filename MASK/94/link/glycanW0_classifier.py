"""
GlycanW0 分类训练脚本 - 单分类任务

使用 GlycanW0 提取的 embedding 训练分类头，配置：
- CrossEntropyLoss (支持类别权重)
- AdamW optimizer
- MultiStageReduceLROnPlateau scheduler
- EMA-based early stopping

单分类任务，标签列：glycantype
过滤频次<30的标签
五个种子：42, 58, 518, 5188, 51888
输出：macro F1, macro AUC, 五次平均混淆矩阵
"""

import os
import json
import random
import sys
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score

SEEDS = list(range(1, 11))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


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


class GlycanW0Classifier(nn.Module):
    def __init__(self, input_dim=1027, hidden_dim=256, num_classes=None, dropout=0.3):
        super().__init__()
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


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, num_classes):
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
    
    probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()
    preds = np.argmax(all_logits, axis=1)

    accuracy = accuracy_score(all_labels, preds)
    f1_macro = f1_score(all_labels, preds, average='macro', zero_division=0)
    
    try:
        macro_auc = float(roc_auc_score(all_labels, probs, multi_class='ovr', average='macro'))
    except ValueError:
        macro_auc = 0.0

    cm = confusion_matrix(all_labels, preds)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'macro_auc': macro_auc,
    }

    return total_loss / num_batches, metrics, preds, all_labels, probs, cm


def load_data(csv_path, label_col='glycantype', min_samples=30):
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")
    
    label_counts = df[label_col].value_counts()
    valid_labels = set(label_counts[label_counts >= min_samples].index)
    
    print(f"Total labels before filtering: {len(label_counts)}")
    print(f"Valid labels (>= {min_samples} samples): {len(valid_labels)}")
    
    for label, count in label_counts.items():
        if count < min_samples:
            print(f"  Filtering out: {label} ({count} samples)")
    
    df = df[df[label_col].isin(valid_labels)].reset_index(drop=True)
    
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col])
    
    embeddings = np.array([json.loads(e) for e in df['embedding']], dtype=np.float32)

    # 读取样本标识列（用于导出分错样本时定位原始糖链）
    glytoucan_ids = df['GlyTouCan ID'].fillna('').astype(str).tolist() if 'GlyTouCan ID' in df.columns else [''] * len(df)
    glycans = df['IUPAC Condensed'].fillna('').astype(str).tolist() if 'IUPAC Condensed' in df.columns else [''] * len(df)
    wurcs_list = df['WURCS'].fillna('').astype(str).tolist() if 'WURCS' in df.columns else [''] * len(df)

    return embeddings, labels, le.classes_, glytoucan_ids, glycans, wurcs_list


def train_with_seed(seed, embeddings, labels, classes, output_dir, glytoucan_ids=None, glycans=None, wurcs_list=None,
                    batch_size=64, hidden_dim=256, dropout=0.3, lr=5e-4, weight_decay=1e-4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_classes = len(classes)
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(splitter.split(embeddings, labels))
    
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(splitter2.split(embeddings[temp_idx], labels[temp_idx]))
    
    train_indices = train_idx.tolist()
    val_indices = temp_idx[val_idx].tolist()
    test_indices = temp_idx[test_idx].tolist()

    X_train, X_val, X_test = embeddings[train_indices], embeddings[val_indices], embeddings[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    test_dataset = EmbeddingDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GlycanW0Classifier(input_dim=1027, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = MultiStageReduceLROnPlateau(optimizer, mode="max", milestones=(18, 36, 54),
                                           factors=(0.5, 1/3, 1/4), min_lr=1e-6)
    early_stopper = EarlyStopping(patience=68, min_delta=1e-4, mode="max", ema_window=5, use_ema=True)

    MAX_EPOCHS = 1000
    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, num_classes)
        val_auc = val_metrics['macro_auc']
        val_acc = val_metrics['accuracy']

        stopped = early_stopper(val_auc)
        ema_auc = early_stopper.ema_value if early_stopper.ema_value is not None else val_auc
        scheduler.step(ema_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())

        print(f"Epoch {epoch:4d}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {val_auc:.4f} | EMA5: {ema_auc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e} | ES: {early_stopper.counter}/{early_stopper.patience}")

        if stopped:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    
    test_loss, test_metrics, test_preds, test_labels, test_probs, test_cm = evaluate(model, test_loader, criterion, num_classes)
    
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "val_macro_auc": best_val_auc,
        "label_classes": classes.tolist(),
    }, os.path.join(seed_dir, "best_model.pt"))

    # 导出错误分析 JSON（测试集中被分错的样本，格式与 GlycanGT 副本的 misclassified_samples.json 类似）
    misclassified_samples = []
    for local_i, global_idx in enumerate(test_indices):
        true_idx = int(test_labels[local_i])
        pred_idx = int(test_preds[local_i])
        if true_idx == pred_idx:
            continue

        prob_dict = {
            str(classes[j]): round(float(test_probs[local_i, j]), 6)
            for j in range(num_classes)
        }

        misclassified_samples.append({
            "row_index": int(global_idx),
            "glytoucan_id": glytoucan_ids[global_idx] if glytoucan_ids is not None and global_idx < len(glytoucan_ids) else None,
            "wurcs": wurcs_list[global_idx] if wurcs_list is not None and global_idx < len(wurcs_list) else None,
            "glycan": glycans[global_idx] if glycans is not None and global_idx < len(glycans) else None,
            "true_label": str(classes[true_idx]),
            "predicted_label": str(classes[pred_idx]),
            "probabilities": prob_dict,
        })

    error_analysis = {
        "seed": seed,
        "description": {
            "scope": "test set only (10% split)",
            "row_index": "过滤低频标签(<min_samples)后数据中的行号，与 embedding 行号一致",
            "probabilities": "各类别的 softmax 概率，键顺序与 class_order 一致",
        },
        "num_test_samples": int(len(test_indices)),
        "num_misclassified": len(misclassified_samples),
        "class_order": classes.tolist(),
        "misclassified_samples": misclassified_samples,
    }

    error_json_path = os.path.join(seed_dir, "misclassified_samples.json")
    with open(error_json_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    print(f"Misclassified samples: {len(misclassified_samples)}/{len(test_indices)}")
    print(f"Error analysis JSON saved to: {error_json_path}")

    result = {
        'seed': seed,
        'f1_macro': test_metrics['f1_macro'],
        'macro_auc': test_metrics['macro_auc'],
        'accuracy': test_metrics['accuracy'],
        'confusion_matrix': test_cm,
        'best_epoch': best_epoch,
    }
    
    return result


def plot_confusion_matrix(cm, classes, save_path, title='Average Confusion Matrix',
                          normalize=True, base_fontsize=18, cell_per_inch=0.85):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm / row_sums
        cm_normalized = np.nan_to_num(cm_normalized)
        display_cm = cm_normalized
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        display_cm = cm
        fmt = '.1f'

    n = len(classes)
    # 画布随类别数自适应，避免字放大后格子被挤扁
    fig_w = max(10, n * cell_per_inch + 3)
    fig_h = max(8, n * cell_per_inch + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(display_cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={'size': base_fontsize, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})

    ax.set_title(title, fontsize=base_fontsize + 8, fontweight='bold', pad=18)
    ax.set_xlabel('Predicted', fontsize=base_fontsize + 4, labelpad=12)
    ax.set_ylabel('True', fontsize=base_fontsize + 4, labelpad=12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                       fontsize=base_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=base_fontsize)

    # 放大 colorbar 刻度字号
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=base_fontsize - 2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default=os.path.join(script_dir, 'train', 'glyconnect_structures_with_embeddings.csv'),
                        help='Input CSV file with embeddings')
    parser.add_argument('--output_dir', type=str, default=os.path.join(script_dir, 'train', 'classification_results'),
                        help='Output directory for results')
    parser.add_argument('--label_col', type=str, default='glycantype',
                        help='Label column name')
    parser.add_argument('--min_samples', type=int, default=30,
                        help='Minimum samples per class')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    log_path = os.path.join(args.output_dir, "training_log.txt")
    sys.stdout = Logger(log_path)

    print(f"\n{'='*60}")
    print(f"GlycanW0 Single-Class Classification")
    print(f"Label column: {args.label_col}")
    print(f"Min samples per class: {args.min_samples}")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*60}")

    print("\nLoading data...")
    embeddings, labels, classes, glytoucan_ids, glycans, wurcs_list = load_data(args.input_csv, args.label_col, args.min_samples)
    num_classes = len(classes)
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(embeddings)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {classes.tolist()}")
    
    class_counts = np.bincount(labels)
    for i, cls in enumerate(classes):
        print(f"  {cls}: {class_counts[i]} samples")

    results = []
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Training with seed: {seed}")
        print(f"{'='*60}")
        
        result = train_with_seed(seed, embeddings, labels, classes, args.output_dir,
                                glytoucan_ids=glytoucan_ids, glycans=glycans, wurcs_list=wurcs_list,
                                batch_size=args.batch_size, hidden_dim=args.hidden_dim,
                                dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay)
        
        results.append(result)
        
        print(f"  Seed {seed} results:")
        print(f"    Macro F1: {result['f1_macro']:.4f}")
        print(f"    Macro AUC: {result['macro_auc']:.4f}")
        print(f"    Accuracy: {result['accuracy']:.4f}")
        print(f"    Best epoch: {result['best_epoch']}")

    print(f"\n{'='*60}")
    print("Summary of all seeds:")
    print(f"{'='*60}")
    
    f1_scores = [r['f1_macro'] for r in results]
    auc_scores = [r['macro_auc'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    print(f"{'Metric':<15} | {'Seed 42':<8} | {'Seed 58':<8} | {'Seed 518':<8} | {'Seed 5188':<9} | {'Seed 51888':<10} | {'Mean':<8} | {'Std':<8}")
    print(f"{'-'*110}")
    print(f"{'Macro F1':<15} | {f1_scores[0]:<8.4f} | {f1_scores[1]:<8.4f} | {f1_scores[2]:<8.4f} | {f1_scores[3]:<9.4f} | {f1_scores[4]:<10.4f} | {np.mean(f1_scores):<8.4f} | {np.std(f1_scores):<8.4f}")
    print(f"{'Macro AUC':<15} | {auc_scores[0]:<8.4f} | {auc_scores[1]:<8.4f} | {auc_scores[2]:<8.4f} | {auc_scores[3]:<9.4f} | {auc_scores[4]:<10.4f} | {np.mean(auc_scores):<8.4f} | {np.std(auc_scores):<8.4f}")
    print(f"{'Accuracy':<15} | {accuracies[0]:<8.4f} | {accuracies[1]:<8.4f} | {accuracies[2]:<8.4f} | {accuracies[3]:<9.4f} | {accuracies[4]:<10.4f} | {np.mean(accuracies):<8.4f} | {np.std(accuracies):<8.4f}")

    avg_cm = np.mean([r['confusion_matrix'] for r in results], axis=0)
    
    row_sums = avg_cm.sum(axis=1, keepdims=True)
    avg_cm_normalized = avg_cm / row_sums
    avg_cm_normalized = np.nan_to_num(avg_cm_normalized)
    
    print(f"\nAverage Confusion Matrix (across {len(SEEDS)} seeds):")
    print("Labels:", classes.tolist())
    print(avg_cm)
    
    print(f"\nNormalized Average Confusion Matrix (row-wise):")
    print(avg_cm_normalized)
    
    np.save(os.path.join(args.output_dir, "average_confusion_matrix.npy"), avg_cm)
    np.save(os.path.join(args.output_dir, "average_confusion_matrix_normalized.npy"), avg_cm_normalized)
    
    plot_confusion_matrix(avg_cm, classes, os.path.join(args.output_dir, "average_confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(avg_cm, classes, os.path.join(args.output_dir, "average_confusion_matrix_normalized.png"), normalize=True)
    print(f"\nConfusion matrix images saved to: {args.output_dir}")

    results_serializable = []
    for r in results:
        item = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                item[k] = v.tolist()
            else:
                item[k] = v
        results_serializable.append(item)
    
    summary = {
        'label_col': args.label_col,
        'min_samples': args.min_samples,
        'num_classes': num_classes,
        'classes': classes.tolist(),
        'seeds': SEEDS,
        'results': results_serializable,
        'mean_f1_macro': float(np.mean(f1_scores)),
        'std_f1_macro': float(np.std(f1_scores)),
        'mean_macro_auc': float(np.mean(auc_scores)),
        'std_macro_auc': float(np.std(auc_scores)),
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'average_confusion_matrix': avg_cm.tolist(),
        'average_confusion_matrix_normalized': avg_cm_normalized.tolist(),
    }
    
    with open(os.path.join(args.output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {os.path.join(args.output_dir, 'summary.json')}")
    print(f"\nAll results saved to: {args.output_dir}")
    
    sys.stdout.log.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()