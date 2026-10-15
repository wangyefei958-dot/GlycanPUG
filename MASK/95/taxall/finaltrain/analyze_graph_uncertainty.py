# -*- coding: utf-8 -*-
"""
图结构不确定性特征 vs 预测对错 分析 (纯CPU, 多进程, 不加载模型)

逻辑:
  1. CSV = 全量样本; 用 MultilabelStratifiedShuffleSplit 重现 test 集
     (只需要标签, 不需要 embedding)
  2. misclassified_samples.json 的 row_index 精确标出 test 里分错的样本
  3. test - misclassified = 分对样本
  4. 用 parser 解析两组样本 WURCS 图, 比较下列特征 (词汇表口径:
     ? / x = 不确定; N/A = 确认没有 = 确定):
       doubt_edge_ratio        : prob<1 的概率边 / 总边数
       mean_edge_prob          : 所有边的平均 probability
       big_edge_uncertainty     : 1 - true_linkage_count/(big_edges/2), 未确认大边连接占比
       small_edge_uncertainty   : 1 - sum(碳位 carbon_count)/实际小边数, doubt 候选位占比
       ring_bond_doubt         : ring_bond_type 含 "?" 的环键占比
       big_edge_doubt          : 大边 "d-a" 中 "?" 计分均值 (1个?=0.5, 2个?=1.0)
       anomer_x                : anomeric_config_type == "x" 的大节点占比
       stereo_x                : stereo_type == "x" 的碳位占比
                                 (按 probability 加权, firm 边 prob=1 全额计入,
                                  prob<1 的 doubt 候选边按其概率计入)
  5. 5 个种子各自统计, 报告 mean +/- std; 画 2x2 特征柱状图
"""
import sys, json, csv, time, pickle, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# 论文用图: 全局放大字体
plt.rcParams.update({
    "font.size":        24,   # 基础字号
    "axes.titlesize":   28,   # 子图标题
    "axes.labelsize":   26,   # 坐标轴标题 (ylabel)
    "xtick.labelsize":  23,   # x 轴刻度
    "ytick.labelsize":  23,   # y 轴刻度
    "legend.fontsize":  22,   # 图例
})

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SEEDS = ["42", "518", "5188", "51888", "58"]
SEED_INT = {"42": 42, "518": 518, "5188": 5188, "51888": 51888, "58": 58}
LEVELS = ["superkingdom", "kingdom", "phylum", "class",
          "order", "family", "genus", "species"]

# (key, 显示名, 是否 doubt 型指标[越高=越不确定])
METRICS = [
    ("doubt_edge_ratio",     "prob<1 edge ratio",        True),
    ("mean_edge_prob",       "mean edge prob",           False),
    ("big_edge_uncertainty",   "big-edge uncertainty",   True),
    ("small_edge_uncertainty", "small-edge uncertainty", True),
    ("ring_bond_doubt",      "ring-bond ? ratio",        True),
    ("big_edge_doubt",       "big-edge ? score",         True),
    ("anomer_x",             "anomeric x ratio",         True),
    ("stereo_x",             "stereo x ratio",           True),
]
METRIC_KEYS = [m[0] for m in METRICS]

# 输出指标: 终端统计表 / summary_table.json / Cohen's d + AUROC 组合图
# 只保留与小图一致的 6 个; doubt_edge_ratio 和 mean_edge_prob 仅参与内部计算
OUT_METRIC_KEYS = ["ring_bond_doubt", "big_edge_doubt", "anomer_x", "stereo_x",
                   "big_edge_uncertainty", "small_edge_uncertainty"]

OUT = ROOT / "uncertainty_graph"
OUT.mkdir(exist_ok=True)
# v2: stereo_x 改为按 probability 加权口径, 与 v1 缓存不兼容
CACHE_PKL = OUT / "_global_wurcs_cache_v2.pkl"


# ============ 1. WURCS 解析 (多进程, 每个唯一 WURCS 只解析一次) ============
def _parse_one(wurcs_str):
    from wurcs_parser_reverse_edge import WURCSParser
    try:
        g = WURCSParser().parse(wurcs_str)
    except Exception:
        return (wurcs_str, None)
    big_p = [e["probability"] for e in g["big_edges"]]
    small_p = [e["probability"] for e in g["small_edges"]]
    all_p = big_p + small_p
    if not all_p:
        return (wurcs_str, None)
    arr = np.array(all_p, dtype=np.float64)

    # 大边确定度 (parser 过滤逻辑同款): true_linkage_count / (big_edges/2)
    n_big_undirected = len(big_p) / 2
    true_linkage = g["metadata"].get("true_linkage_count", 0)
    big_edge_certainty = (true_linkage / n_big_undirected) if n_big_undirected > 0 else 1.0

    # 小边确定度: 真实碳位连接数 / 实际小边数 (doubt 候选边使分母膨胀)
    true_small, carbon_ok = 0, True
    for bn in g["big_nodes"]:
        cc = bn.get("carbon_count_type")
        if isinstance(cc, str) and cc.isdigit():
            true_small += int(cc)
        else:
            carbon_ok = False
            break
    small_edge_certainty = (true_small / len(small_p)) if (carbon_ok and small_p) else None

    # ---- 特征级不确定性 (? / x = 不确定; N/A = 确定没有) ----
    # ring_bond_type 含 "?" 的环键占比
    rb_all = [rb for bn in g["big_nodes"] for rb in bn.get("ring_bonds", [])]
    ring_bond_doubt = (sum(1 for rb in rb_all if "?" in rb["ring_bond_type"]) / len(rb_all)
                       if rb_all else None)

    # 大边 "?" 计分: 无向去重, "d-a" 部分, 1个?=0.5, 2个?=1.0
    seen, e_scores = set(), []
    for e in g["big_edges"]:
        key = (min(e["source_big_node_id"], e["target_big_node_id"]),
               max(e["source_big_node_id"], e["target_big_node_id"]))
        if key in seen:
            continue
        seen.add(key)
        e_scores.append(e["big_edge_type"].split("*")[0].count("?") / 2.0)
    big_edge_doubt = float(np.mean(e_scores)) if e_scores else None

    # anomeric_config_type == "x" 的大节点占比 (N/A = 确定)
    n_bn = len(g["big_nodes"])
    anomer_x = (sum(1 for bn in g["big_nodes"] if bn.get("anomeric_config_type") == "x") / n_bn
                if n_bn else None)

    # stereo_type == "x" 的碳位占比: 按 probability 加权
    # firm 边 (prob=1) 全额计入; prob<1 的 doubt 候选边按其概率计入
    # (同一 doubt 取代基在各候选位的概率和为 1, 不会重复计数)
    w_num, w_den = 0.0, 0.0
    for se in g["small_edges"]:
        if "stereo_type" not in se:
            continue
        p = float(se["probability"])
        if p <= 0.0:
            continue
        w_den += p
        if se["stereo_type"] == "x":
            w_num += p
    stereo_x = (w_num / w_den) if w_den > 0 else None

    return (wurcs_str, {
        "mean_edge_prob":       float(arr.mean()),
        "doubt_edge_ratio":     float((arr < 1.0).mean()),
        "big_edge_certainty":   float(big_edge_certainty),
        "small_edge_certainty": small_edge_certainty,
        "ring_bond_doubt":      ring_bond_doubt,
        "big_edge_doubt":       big_edge_doubt,
        "anomer_x":             anomer_x,
        "stereo_x":             stereo_x,
    })


def build_global_cache():
    if CACHE_PKL.exists():
        print(f"[cache hit] {CACHE_PKL.name}")
        with open(CACHE_PKL, "rb") as f:
            return pickle.load(f)

    all_wurcs = set()
    for seed in SEEDS:
        for level in LEVELS:
            p = ROOT / seed / "split_files_processed_wurcs" / f"{level}.csv"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("WURCS"):
                        all_wurcs.add(row["WURCS"])
    print(f"Unique WURCS: {len(all_wurcs):,}")

    wlist = list(all_wurcs)
    nw = max(1, min(multiprocessing.cpu_count() - 1, 12))
    print(f"Parsing with {nw} workers...")
    t0 = time.time()
    cache, done = {}, 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futs = [pool.submit(_parse_one, w) for w in wlist]
        for fut in as_completed(futs):
            w, feat = fut.result()
            cache[w] = feat
            done += 1
            if done % 4000 == 0 or done == len(wlist):
                print(f"  {done}/{len(wlist)} ({done/(time.time()-t0):.0f}/s)")
    print(f"Parsed in {time.time()-t0:.1f}s")
    with open(CACHE_PKL, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


# ============ 2. 单个 (seed, level) worker ============
def _process_one_task(args):
    """返回 (seed, level, {metric: {ok: arr, wrong: arr}})"""
    seed, level = args
    base = ROOT / seed / "split_files_processed_wurcs"
    csv_path = base / f"{level}.csv"
    miss_path = base / f"training_results_{level}" / "misclassified_samples.json"

    global _WURCS_CACHE
    if "_WURCS_CACHE" not in globals():
        with open(CACHE_PKL, "rb") as f:
            _WURCS_CACHE = pickle.load(f)
    cache = _WURCS_CACHE

    wurcs_list, labels_raw = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wurcs_list.append(row.get("WURCS", ""))
            labels_raw.append(row.get(level, ""))

    uniq = sorted({p for ls in labels_raw
                   for p in (ls.split("_") if "_" in ls else [ls])})
    l2i = {l: i for i, l in enumerate(uniq)}
    y = np.zeros((len(labels_raw), len(uniq)), dtype=np.float32)
    for i, ls in enumerate(labels_raw):
        for p in (ls.split("_") if "_" in ls else [ls]):
            if p in l2i:
                y[i, l2i[p]] = 1.0

    # 重现 test split
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    X_dummy = np.zeros((len(y), 1), dtype=np.float32)
    rs = SEED_INT[seed]
    sp1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    _, temp_idx = next(sp1.split(X_dummy, y))
    sp2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=rs)
    _, ti = next(sp2.split(X_dummy[temp_idx], y[temp_idx]))
    test_indices = temp_idx[ti].tolist()

    with open(miss_path, encoding="utf-8") as f:
        miss = json.load(f)
    wrong_rows = {s["row_index"] for s in miss.get("misclassified_samples", [])}

    ok = {m: [] for m in METRIC_KEYS}
    wrong = {m: [] for m in METRIC_KEYS}
    n_no_feat = 0
    for ri in test_indices:
        feat = cache.get(wurcs_list[ri])
        if feat is None:
            n_no_feat += 1
            continue
        # 统一不确定性口径: 缓存里存的是 certainty, 读出后翻转为 1 - certainty
        # (翻转只影响展示方向, MW p / Cohen d / AUROC 数值不变)
        feat = dict(feat)
        for ckey, ukey in (("big_edge_certainty", "big_edge_uncertainty"),
                           ("small_edge_certainty", "small_edge_uncertainty")):
            if feat.get(ckey) is not None:
                feat[ukey] = 1.0 - feat[ckey]
        tgt = wrong if ri in wrong_rows else ok
        for m in METRIC_KEYS:
            v = feat.get(m)
            if v is not None:
                tgt[m].append(v)

    return (seed, level, {
        "ok":    {m: np.array(v) for m, v in ok.items()},
        "wrong": {m: np.array(v) for m, v in wrong.items()},
        "n_no_feat": n_no_feat,
        "n_test": len(test_indices),
    })


# ============ 3. 主流程 ============
def main():
    t0 = time.time()
    build_global_cache()

    tasks = [(s, lv) for lv in LEVELS for s in SEEDS]
    results = {}
    print(f"\nProcessing {len(tasks)} (seed x level) tasks in parallel...")
    with ProcessPoolExecutor(max_workers=min(10, len(tasks))) as pool:
        futs = [pool.submit(_process_one_task, t) for t in tasks]
        for fut in as_completed(futs):
            seed, level, r = fut.result()
            results[(seed, level)] = r
            print(f"  done: seed={seed:<6} {level:<13} "
                  f"test={r['n_test']}  ok={len(r['ok']['doubt_edge_ratio'])}  "
                  f"wrong={len(r['wrong']['doubt_edge_ratio'])}")

    # ---------- 统计表: 每个 (level, metric) 一行, 5 种子平均 ----------
    # 新增 Cohen's d 和 AUROC, 区分"均值接近"到底是:
    #   (a) 真没区分力 (AUROC≈0.5, |d|<0.2)  -- 指标不重要
    #   (b) 有阈值效应 / 分布重叠复杂 (AUROC 高但均值接近) -- 小差异就影响模型
    print(f"\n{'='*130}")
    print("5-seed mean +/- std;  MW = pooled Mann-Whitney;  d = Cohen's d (direction-aligned);  AUC = single-feature AUROC")
    print(f"{'='*130}")
    print(f"{'Level':<13} | {'metric':<21} | {'OK mean':>9} {'WR mean':>9} | "
          f"{'OK std':>7} {'WR std':>7} | {'MW_p':>9} | {'Cohen d':>8} {'AUC':>7}")
    print("-" * 130)

    table = {}
    for level in LEVELS:
        seed_rows = [results[(s, level)] for s in SEEDS
                     if (s, level) in results
                     and len(results[(s, level)]["ok"]["doubt_edge_ratio"]) > 0
                     and len(results[(s, level)]["wrong"]["doubt_edge_ratio"]) > 0]
        if not seed_rows:
            continue
        table[level] = {}
        for mkey, mname, is_doubt in METRICS:
            if mkey not in OUT_METRIC_KEYS:
                continue
            ok_arrays = [r["ok"][mkey] for r in seed_rows
                         if len(r["ok"][mkey]) > 0]
            wr_arrays = [r["wrong"][mkey] for r in seed_rows
                         if len(r["wrong"][mkey]) > 0]
            if not ok_arrays or not wr_arrays:
                continue
            ok_pooled = np.concatenate(ok_arrays)
            wr_pooled = np.concatenate(wr_arrays)
            ok_mu, ok_sd = ok_pooled.mean(), ok_pooled.std(ddof=1)
            wr_mu, wr_sd = wr_pooled.mean(), wr_pooled.std(ddof=1)

            # Mann-Whitney: doubt 型 wrong>ok, certainty 型 ok>wrong
            if is_doubt:
                _, p = mannwhitneyu(wr_pooled, ok_pooled, alternative="greater")
            else:
                _, p = mannwhitneyu(ok_pooled, wr_pooled, alternative="greater")

            # Cohen's d: 标准化效应量, 按指标类型统一方向:
            #   doubt 型     -> (wrong - ok) / pooled_sd
            #   certainty 型 -> (ok - wrong) / pooled_sd
            # => 正值 = 差异方向符合预期, |d| 越大区分力越强
            # pooled_sd 用经典 df 加权口径 (d_s):
            #   sqrt(((n_ok-1)*s_ok^2 + (n_wr-1)*s_wr^2) / (n_ok+n_wr-2))
            n_ok, n_wr = len(ok_pooled), len(wr_pooled)
            df = n_ok + n_wr - 2
            pooled_sd = (np.sqrt(((n_ok - 1) * ok_sd**2 + (n_wr - 1) * wr_sd**2) / df)
                         if df > 0 else 0.0)
            diff = (wr_mu - ok_mu) if is_doubt else (ok_mu - wr_mu)
            cohen_d = diff / pooled_sd if pooled_sd > 0 else 0.0

            # 单特征 AUROC: 把该指标当分类器, 预测 correct (1) vs wrong (0)
            # doubt 型指标越大越可能 wrong, 直接用; certainty 型取反 (1-x)
            if is_doubt:
                scores = np.concatenate([ok_pooled, wr_pooled])
            else:
                scores = np.concatenate([1.0 - ok_pooled, 1.0 - wr_pooled])
            labels = np.concatenate([np.zeros_like(ok_pooled),
                                     np.ones_like(wr_pooled)])
            # 手动算 AUC (避免 sklearn 依赖, 且更稳健)
            order = np.argsort(scores)
            labels_sorted = labels[order]
            n_pos = labels_sorted.sum()
            n_neg = len(labels_sorted) - n_pos
            if n_pos == 0 or n_neg == 0:
                auroc = 0.5
            else:
                ranks = np.empty(len(labels_sorted))
                ranks[order] = np.arange(1, len(labels_sorted) + 1)
                # 处理同分: 取平均秩
                unique_scores, inv_idx, counts = np.unique(scores,
                                                           return_inverse=True,
                                                           return_counts=True)
                for i, c in enumerate(counts):
                    if c > 1:
                        avg_rank = ranks[inv_idx == i].mean()
                        ranks[inv_idx == i] = avg_rank
                # ranks 按原始样本顺序索引, 掩码必须用原始顺序的 labels
                sum_pos_ranks = ranks[labels == 1].sum()
                auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                auroc = float(np.clip(auroc, 0.0, 1.0))

            print(f"{level:<13} | {mname:<21} | {ok_mu:>9.4f} {wr_mu:>9.4f} | "
                  f"{ok_sd:>7.4f} {wr_sd:>7.4f} | {p:>9.2e} | {cohen_d:>8.3f} {auroc:>7.3f}")
            table[level][mkey] = {
                "ok_mean": float(ok_mu), "ok_std": float(ok_sd),
                "wrong_mean": float(wr_mu), "wrong_std": float(wr_sd),
                "mannwhitney_p": float(p),
                "cohen_d": float(cohen_d),
                "auroc": float(auroc),
            }
        print("-" * 130)

    with open(OUT / "summary_table.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    # ---------- 逐个特征出图 ----------
    all_metrics = {
        "ring_bond_doubt":     ("Ring bond type ? ratio",                     None),
        "big_edge_doubt":     ("Residue edge type ? score (1?=0.5, 2?=1.0)",  None),
        "anomer_x":            ("Anomeric 'x' ratio",                         None),
        "stereo_x":            ("Carbon descriptor 'x' weighted ratio",       None),
        "big_edge_uncertainty":  ("Inter residue edge uncertainty",   "1 - true_linkage / (big_edges/2)"),
        "small_edge_uncertainty":("Substituent attachment edge uncertainty", "1 - sum(carbon positions) / n_small_edges"),
    }
    x = np.arange(len(LEVELS))
    w = 0.38

    def _panel(ax, mkey):
        title, ylabel = all_metrics[mkey]
        ok_h = [table[lv][mkey]["ok_mean"] for lv in LEVELS if mkey in table[lv]]
        wr_h = [table[lv][mkey]["wrong_mean"] for lv in LEVELS if mkey in table[lv]]
        ok_e = [table[lv][mkey]["ok_std"] for lv in LEVELS if mkey in table[lv]]
        wr_e = [table[lv][mkey]["wrong_std"] for lv in LEVELS if mkey in table[lv]]
        err_kw = {"alpha": 0.35, "elinewidth": 1.2}
        b1 = ax.bar(x - w/2, ok_h, w, yerr=ok_e, capsize=3,
                    color="#4CAF50", alpha=0.85, label="Correct", error_kw=err_kw)
        b2 = ax.bar(x + w/2, wr_h, w, yerr=wr_e, capsize=3,
                    color="#F44336", alpha=0.85, label="Misclassified", error_kw=err_kw)
        for bars in (b1, b2):
            for b in bars:
                ax.annotate(f"{b.get_height():.3f}",
                            (b.get_x() + b.get_width()/2, b.get_height()),
                            ha="center", va="bottom", fontsize=16, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS, rotation=30, ha="right")
        ax.set_title(title, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)  # y 轴从 0 开始, 避免底部空隙

    out_pngs = []
    for mkey in all_metrics:
        if not any(mkey in table[lv] for lv in LEVELS):
            continue
        fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
        _panel(ax, mkey)
        out_png = OUT / f"bar_{mkey}.png"
        plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
        plt.close()
        out_pngs.append(out_png)

    # ---------- Cohen's d + AUROC 组合图 (上下两面板, 共享层级横坐标) ----------
    # 上面板: |Cohen's d| (方向对齐后取绝对值, 纯区分强度, 0 = 无区分力)
    # 下面板: 单特征 AUROC, 以 0.5 (随机) 为基线, 柱越向下伸区分力越强
    out_metrics = [(mk, mn, idt) for mk, mn, idt in METRICS
                   if mk in OUT_METRIC_KEYS]
    n_m = len(out_metrics)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_m)]
    x = np.arange(len(LEVELS))
    bw = 0.8 / n_m
    short_names = {
        "big_edge_uncertainty":   "inter residue edge",
        "small_edge_uncertainty": "substituent attachment edge",
        "ring_bond_doubt":      "ring bond type",
        "big_edge_doubt":       "residue edge type",
        "anomer_x":             "anomeric config",
        "stereo_x":             "Carbon descriptor",
    }

    fig, (ax_d, ax_a) = plt.subplots(2, 1, figsize=(20, 13.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1.25, 1.0]})
    # 手动布局: 两面板左缘严格对齐 (纵轴成一条直线), 面板间距收紧
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.03, hspace=0.14)
    ax_d.tick_params(axis="x", length=0)  # 去掉上面板底部残留刻度线, 缝隙更干净
    for i, (mkey, _, _) in enumerate(out_metrics):
        lv_idx = [j for j, lv in enumerate(LEVELS) if mkey in table[lv]]
        if not lv_idx:
            continue
        d_vals = [abs(table[LEVELS[j]][mkey]["cohen_d"]) for j in lv_idx]
        a_vals = [table[LEVELS[j]][mkey]["auroc"] for j in lv_idx]
        off = (i - n_m / 2 + 0.5) * bw
        ax_d.bar(np.array(lv_idx) + off, d_vals, bw * 0.92,
                 color=colors[i], label=short_names[mkey])
        # AUROC 以 0.5 (随机) 为基线: >=0.5 向下伸, <0.5 (方向未对齐) 向上伸
        ax_a.bar(np.array(lv_idx) + off, [abs(v - 0.5) for v in a_vals],
                 bw * 0.92, bottom=[min(v, 0.5) for v in a_vals], color=colors[i])
        for j, v in zip(lv_idx, d_vals):
            ax_d.annotate(f"{v:.2f}", (j + off, v), ha="center",
                          va="bottom", fontsize=12, rotation=90, zorder=5)
        for j, v in zip(lv_idx, a_vals):
            # 数字统一放在柱尖下方 (面板纵轴反写: 向下 = 远离 0.5 基线)
            ax_a.annotate(f"{v:.2f}", (j + off, v),
                          ha="center", va="top",
                          fontsize=12, rotation=90, zorder=5)

    ax_d.set_ylabel("|Cohen's d|")
    all_d = [abs(table[lv][mk]["cohen_d"]) for lv in LEVELS
             for mk in OUT_METRIC_KEYS if mk in table[lv]]
    d_top = (max(all_d) * 1.18) if all_d else 0.8
    ax_d.set_ylim(0, d_top)  # 绝对值口径, 基线 0 = 无区分力; 顶部留余量防数值被裁
    ax_d.set_title("Graph Uncertainty: Correct vs. Wrong",
                   fontweight="bold")
    ax_d.legend(ncol=2, loc="upper right")  # 6 项 x 2 列 = 3 行, 窄图例不挡柱子
    ax_d.grid(axis="y", alpha=0.3)

    all_a = [table[lv][mk]["auroc"] for lv in LEVELS
             for mk in OUT_METRIC_KEYS if mk in table[lv]]
    y_top = min(1.0, (max(all_a) + 0.05)) if all_a else 0.6
    # 基线 0.5 之上留空间: AUROC<0.5 的柱子朝上伸
    y_lo = 0.48 if all_a and min(all_a) < 0.5 else 0.5
    ax_a.set_ylim(y_top, y_lo)  # 纵坐标反写: 0.5 (随机) 在顶部, 越往下区分力越强
    ax_a.set_ylabel("AUROC (0.5 = chance)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(LEVELS, rotation=0, ha="center")  # 水平放置, 不倾斜
    ax_a.xaxis.set_ticks_position("top")    # 横坐标刻度移到两面板之间 (下面板顶边)
    ax_a.xaxis.set_label_position("top")
    ax_a.grid(axis="y", alpha=0.3)

    out_comb = OUT / "bar_cohen_d_auroc.png"
    plt.savefig(str(out_comb), dpi=150, bbox_inches="tight")
    plt.close()
    out_pngs.append(out_comb)

    print(f"\nFigures ({len(out_pngs)}):")
    for p in out_pngs:
        print(f"  {p}")
    print(f"Table : {OUT / 'summary_table.json'}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
