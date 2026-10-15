"""
Get glycan embeddings from IUPAC condensed strings using GlycanGT.

Pipeline:
  (1) Read IUPAC (CSV/TXT) from --input_dir
  (2) Convert to TokenGT graph triples (temporary .json per glycan)
  (3) Load pretrained GlycanGT and extract [graph] embeddings
  (4) Save embeddings to --output_dir (NPY, CSV, and manifest)

Requirements:
  pip install GlycanGT (the HF repo provides the package after `pip install .`)
  and ensure vocab files exist (see --mono_vocab, --link_vocab).

新增功能:
  支持处理 split_files 格式的 CSV（wurcs, level, glycan），在原文件后追加 embedding 列
  使用 --mode split_files 启用此模式
"""

import argparse
import csv
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# -------------------------
# Imports from the package
# -------------------------
try:
    from tokenizer.monomer_vocab import MonomerVocab
    from tokenizer.linkage_vocab import LinkageVocab
    from tokenizer.encode_glycan import iupac_to_graph_triples

    from model.utils import GlycanGraphDataset, collate_fn_for_tokengt, load_vocabs
    from model.glycan_tokengt_multitask import GlycanTokenGTMultiTask
    from model.config_tokengt import get_config
except Exception as e:
    print("[ERROR] Failed to import GlycanGT package modules. "
          "Make sure you've run `pip install .` in the cloned HF repo.")
    raise e


# -------------------------
# Constants / Defaults
# -------------------------
# 项目根目录（本脚本所在目录），所有相对路径一律基于它解析，
# 避免受运行时工作目录（CWD）影响
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_IUPAC_COL = "IUPAC Condensed"
DEFAULT_ID_COL = "GlyTouCan ID"
CKPT_PATTERN = "pretrained_tokengt_{size}_multitask_mask_35_final.pt"
DEFAULT_UNK_TOKEN = "[UNK]"

# -------------------------
# Utilities
# -------------------------
def find_input_files(input_dir: Path, allow_csv=True, allow_txt=True) -> List[Path]:
    files = []
    if allow_csv:
        files += list(input_dir.glob("*.csv"))
    if allow_txt:
        files += list(input_dir.glob("*.txt"))
    return sorted(files)


def load_iupac_from_csv(path: Path, iupac_col: str, id_col: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(path, dtype=str)
    if iupac_col not in df.columns:
        raise ValueError(f"[{path}] column '{iupac_col}' not found.")
    if id_col not in df.columns:
        ids = [f"glycan_{i:06d}" for i in range(len(df))]
    else:
        ids = df[id_col].fillna("").replace("", np.nan)
        ids = ids.fillna(pd.Series([f"glycan_{i:06d}" for i in range(len(df))]))
        ids = ids.astype(str).tolist()

    iupacs = df[iupac_col].astype(str).tolist()
    return list(zip(ids, iupacs))


def load_iupac_from_txt(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            gid = f"glycan_{i:06d}"
            pairs.append((gid, s))
    return pairs


def encode_to_graph_json(
    pairs: List[Tuple[str, str]],
    mono_vocab_path: Path,
    link_vocab_path: Path,
    out_dir: Path,
    id_col: str = DEFAULT_ID_COL,
    iupac_col: str = DEFAULT_IUPAC_COL,
    unk_token: str = DEFAULT_UNK_TOKEN,              
    on_unknown: str = "unk",              
) -> Tuple[List[str], Path]:
    """Convert IUPAC strings to TokenGT-style graph JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mono_vocab = MonomerVocab.load(str(mono_vocab_path))
    link_vocab = LinkageVocab.load(str(link_vocab_path))

    if hasattr(mono_vocab, "add_token") and unk_token not in getattr(mono_vocab, "tokens", []):
        try:
            mono_vocab.add_token(unk_token)
        except Exception:
            pass
    if hasattr(link_vocab, "add_token") and unk_token not in getattr(link_vocab, "tokens", []):
        try:
            link_vocab.add_token(unk_token)
        except Exception:
            pass

    manifest: List[str] = []
    graph_ids: List[str] = []

    for i, (gid, iupac) in enumerate(tqdm(pairs, desc="IUPAC→Graph(JSON)")):
        try:
            triples = iupac_to_graph_triples(
                iupac,
                mono_vocab,
                link_vocab,
                unk_token=unk_token,        
                on_unknown=on_unknown,      
            )
        except TypeError:
            try:
                triples = iupac_to_graph_triples(iupac, mono_vocab, link_vocab)
            except Exception as e:
                if on_unknown == "skip":
                    print(f"[Warn] skip {gid}: {e}")
                    continue
                elif on_unknown == "unk":
                    print(f"[Warn] (fallback=UNK→skip) {gid}: {e}")
                    continue
                else:
                    raise
        except Exception as e:
            if on_unknown == "skip":
                print(f"[Warn] skip {gid}: {e}")
                continue
            elif on_unknown == "unk":
                print(f"[Warn] (fallback=UNK→skip) {gid}: {e}")
                continue
            else:
                raise

        out_path = out_dir / f"{gid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)

        manifest.append(str(out_path))
        graph_ids.append(gid)

    manifest_path = out_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest), encoding="utf-8")
    return graph_ids, manifest_path


def ensure_special_tokens(vocabs: Tuple[List[str], List[str]], unk: str = DEFAULT_UNK_TOKEN) -> Tuple[List[str], List[str]]:
    mono, link = vocabs
    if unk not in mono:
        mono = list(mono) + [unk]
    if unk not in link:
        link = list(link) + [unk]
    if "[MASK]" not in mono:
        mono = list(mono) + ["[MASK]"]
    if "[MASK]" not in link:
        link = list(link) + ["[MASK]"]
    return mono, link


def load_pretrained(size: str, vocabs: Tuple[List[str], List[str]], ckpt_dir: Path, device: torch.device):
    monomer_vocab, linkage_vocab = vocabs
    cfg = get_config(size)
    cfg.update({
        "num_atoms": len(monomer_vocab),
        "num_edges": len(linkage_vocab),
        "orf_node_id": True,
        "lap_node_id": False
    })

    model = GlycanTokenGTMultiTask(cfg=cfg)
    ckpt_path = ckpt_dir / CKPT_PATTERN.format(size=size)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Load] {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    # 静默移动到设备，不打印设备信息
    model.to(device)
    model.eval()
    return model


def extract_graph_embeddings(
    model: torch.nn.Module,
    graph_json_dir: Path,
    graph_ids: List[str],
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
):
    file_paths = [graph_json_dir / f"{gid}.json" for gid in graph_ids]
    dataset = GlycanGraphDataset([str(p) for p in file_paths])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_for_tokengt,
        num_workers=num_workers,
        pin_memory=True
    )

    all_vecs = []
    kept_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract embeddings"):
            if batch is None:
                continue
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            _, extra = model(batch)
            reps = extra["graph_rep"].detach().cpu().numpy()
            all_vecs.append(reps)

    if not all_vecs:
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(all_vecs, axis=0)


def save_outputs(
    out_dir: Path,
    graph_ids: List[str],
    embeddings: np.ndarray,
    prefix: str = "embeddings",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{prefix}.npy"
    np.save(npy_path, embeddings)
    
    cols = [f"dim_{i}" for i in range(embeddings.shape[1])] if embeddings.size else []
    df = pd.DataFrame(embeddings, index=graph_ids, columns=cols)
    df.index.name = "graph_id"
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path)

    (out_dir / f"{prefix}_ids.txt").write_text("\n".join(graph_ids), encoding="utf-8")

    print(f"[Save] NPY: {npy_path}")
    print(f"[Save] CSV: {csv_path}")
    print(f"[Save] IDs: {out_dir / f'{prefix}_ids.txt'}")


# -------------------------
# Split Files Mode Functions
# -------------------------
def process_split_files_mode(
    input_dir: Path,
    output_dir: Path,
    mono_vocab_path: Path,
    link_vocab_path: Path,
    ckpt_dir: Path,
    model_size: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    处理 split_files 模式：读取 CSV 文件，添加 embedding 列，保存到输出目录。
    
    输入格式: wurcs, level, glycan
    输出格式: wurcs, level, glycan, embedding
    """
    print("\n" + "="*60)
    print("Split Files Mode: Processing CSV files")
    print("="*60)

    # 加载预训练模型
    print("\nLoading GlycanGT model...")
    # 使用 load_vocabs 函数加载词汇表（传入 data 目录，函数会自动找 vocab_expanded 子目录）
    data_dir = str(mono_vocab_path.parent.parent)  # 获取 data 目录路径
    monomer_vocab, linkage_vocab = load_vocabs(data_dir)
    vocabs = (monomer_vocab, linkage_vocab)
    model = load_pretrained(model_size, vocabs, ckpt_dir, device)

    # 遍历输入目录下的所有 CSV 文件
    csv_files = sorted(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"[Error] No CSV files found in {input_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process")

    # 处理每个 CSV 文件
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        
        # 读取 CSV
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"  Original rows: {len(df)}")
        
        # 获取 glycan 列数据
        glycans = df['glycan'].astype(str).tolist()
        
        # 创建临时目录存储 graph JSON
        temp_graph_dir = Path(tempfile.mkdtemp())
        
        try:
            # 转换为 graph JSON
            pairs = [(f"glycan_{i:06d}", glycan) for i, glycan in enumerate(glycans)]
            graph_ids = encode_to_graph_json(
                pairs,
                mono_vocab_path,
                link_vocab_path,
                temp_graph_dir,
                unk_token=DEFAULT_UNK_TOKEN,
                on_unknown="skip",
            )[0]
            print(f"  Valid glycans: {len(graph_ids)}")
            
            if len(graph_ids) == 0:
                print(f"  [Warn] No valid glycans found, copying original file")
                df['embedding'] = ''
                output_path = output_dir / csv_file.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False, encoding='utf-8')
                continue
            
            # 提取 embedding
            embeddings = extract_graph_embeddings(
                model, temp_graph_dir, graph_ids, device, 
                batch_size=batch_size, num_workers=num_workers
            )
            print(f"  Extracted embeddings: {len(embeddings)}, shape: {embeddings.shape}")
            
            # 创建 embedding 字典
            id_to_embedding = {gid: json.dumps(emb.tolist()) 
                              for gid, emb in zip(graph_ids, embeddings)}
            
            # 添加 embedding 列到 DataFrame
            embedding_list = []
            for i in range(len(glycans)):
                gid = f"glycan_{i:06d}"
                embedding_list.append(id_to_embedding.get(gid, ''))
            
            df['embedding'] = embedding_list
            
            # 保存处理后的 CSV
            output_path = output_dir / csv_file.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"  Saved to: {output_path}")
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_graph_dir, ignore_errors=True)

    print("\n" + "="*60)
    print("Split Files Mode completed!")
    print(f"Output directory: {output_dir}")
    print("="*60)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Get GlycanGT embeddings from IUPAC condensed strings.")
    
    # 默认模式为 split_files，默认处理 ./split_files 目录
    ap.add_argument("--mode", type=str, default="split_files", 
                    choices=["standard", "split_files"],
                    help="Mode: 'standard' for IUPAC CSV/TXT input, 'split_files' for processing split_files format (wurcs, level, glycan)")
    
    ap.add_argument("--input_dir", type=str, default="./split_files",
                    help="Directory containing input .csv and/or .txt with IUPAC.")
    ap.add_argument("--output_dir", type=str, default="./split_files_processed",
                    help="Directory to write embeddings and logs.")
    ap.add_argument("--mono_vocab", type=str, default="data/vocab_expanded/monomer.json",
                    help="Path to monomer vocab JSON.")
    ap.add_argument("--link_vocab", type=str, default="data/vocab_expanded/linkage.json",
                    help="Path to linkage vocab JSON.")
    ap.add_argument("--checkpoint_dir", type=str, default="outputs/mask_35",
                    help="Directory containing pretrained checkpoints.")
    ap.add_argument("--model_size", type=str, default="large", choices=["ss", "small", "medium", "large"],
                    help="Model scale to load.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Default: auto")
    ap.add_argument("--iupac_col", type=str, default=DEFAULT_IUPAC_COL,
                    help="CSV column name for IUPAC condensed.")
    ap.add_argument("--id_col", type=str, default=DEFAULT_ID_COL,
                    help="CSV column name for glycan ID.")
    ap.add_argument("--keep_graphs", action="store_true",
                    help="Keep temporary graph JSON files.")
    ap.add_argument("--unk_token", type=str, default=DEFAULT_UNK_TOKEN,
                    help="Unknown token symbol to use when a term is not in vocab.")
    ap.add_argument("--on_unknown", type=str, default="skip", choices=["unk", "skip", "error"],
                    help="How to handle unknown terms during IUPAC→graph.")
    
    args = ap.parse_args()

    # 将相对路径统一解析为基于 PROJECT_ROOT 的绝对路径：
    # 相对路径默认按 CWD 解析，从其他目录运行脚本时会找不到文件
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    args.input_dir = str(_resolve(args.input_dir))
    args.output_dir = str(_resolve(args.output_dir))
    args.mono_vocab = str(_resolve(args.mono_vocab))
    args.link_vocab = str(_resolve(args.link_vocab))
    args.checkpoint_dir = str(_resolve(args.checkpoint_dir))

    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # 默认处理 split_files 模式
    if args.mode == "split_files":
        process_split_files_mode(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            mono_vocab_path=Path(args.mono_vocab),
            link_vocab_path=Path(args.link_vocab),
            ckpt_dir=Path(args.checkpoint_dir),
            model_size=args.model_size,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return

    # ==================== 标准模式 ====================
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    tmp_graph_dir = output_dir / "tmp_graphs"

    if not input_dir.exists():
        raise FileNotFoundError(f"--input_dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Gather input files
    files = find_input_files(input_dir, allow_csv=True, allow_txt=True)
    if not files:
        raise FileNotFoundError(f"No .csv or .txt found under {input_dir}")

    # Read IUPACs
    pairs_all: List[Tuple[str, str]] = []
    for f in files:
        if f.suffix.lower() == ".csv":
            pairs = load_iupac_from_csv(f, args.iupac_col, args.id_col)
        elif f.suffix.lower() == ".txt":
            pairs = load_iupac_from_txt(f)
        else:
            continue
        pairs_all.extend(pairs)

    if not pairs_all:
        print("[Warn] No valid IUPAC rows found. Exiting.")
        return

    # IUPAC -> graph JSON
    graph_ids, manifest_path = encode_to_graph_json(
        pairs_all,
        mono_vocab_path=Path(args.mono_vocab),
        link_vocab_path=Path(args.link_vocab),
        out_dir=tmp_graph_dir,
        id_col=args.id_col,
        iupac_col=args.iupac_col,
        unk_token=args.unk_token, 
        on_unknown=args.on_unknown,
    )
    if not graph_ids:
        print("[Warn] No graphs were generated. Exiting.")
        return

    # Load vocabs for model
    monomer_vocab, linkage_vocab = load_vocabs(str(Path(args.mono_vocab).parent.parent)) \
        if "load_vocabs" in globals() else ([], [])
    if not monomer_vocab or not linkage_vocab:
        monomer_vocab = json.loads(Path(args.mono_vocab).read_text(encoding="utf-8"))
        linkage_vocab = json.loads(Path(args.link_vocab).read_text(encoding="utf-8"))
    monomer_vocab, linkage_vocab = ensure_special_tokens((monomer_vocab, linkage_vocab), unk=args.unk_token)

    # Load model
    model = load_pretrained(
        size=args.model_size,
        vocabs=(monomer_vocab, linkage_vocab),
        ckpt_dir=ckpt_dir,
        device=device,
    )

    # Build dataset & extract embeddings
    embeddings = extract_graph_embeddings(
        model=model,
        graph_json_dir=tmp_graph_dir,
        graph_ids=graph_ids,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Save outputs
    save_outputs(
        out_dir=output_dir,
        graph_ids=graph_ids,
        embeddings=embeddings,
        prefix=f"embeddings_{args.model_size}",
    )

    # Cleanup temp graphs
    if not args.keep_graphs:
        shutil.rmtree(tmp_graph_dir, ignore_errors=True)
        print(f"[Clean] Removed temporary graphs: {tmp_graph_dir}")
    else:
        print(f"[Keep] Temporary graphs kept at: {tmp_graph_dir}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
