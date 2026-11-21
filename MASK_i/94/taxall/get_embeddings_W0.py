"""
Get glycan embeddings from WURCS strings using the GNN model.

Pipeline:
  (1) Read WURCS (CSV/TXT) from --input_dir
  (2) Convert to GNN graph format using WURCSParser
  (3) Load pretrained model and extract [graph] embeddings (1027-dim)
  (4) Save embeddings to --output_dir (NPY, CSV, and manifest)

Output embedding dimension: 1027
  - big_node mean+max pooled: 512
  - small_node mean+max pooled: 512
  - global features: 3

Requirements:
  Ensure the WURCSParser is available and vocab files exist.
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

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

# -------------------------
# Local imports
# -------------------------
try:
    from wurcs_parser_reverse_edge import WURCSParser
except ImportError:
    print("[ERROR] Failed to import WURCSParser. Make sure wurcs_parser_reverse_edge.py is in the same directory.")
    raise

# -------------------------
# Constants / Defaults
# -------------------------
DEFAULT_WURCS_COL = "WURCS"
DEFAULT_ID_COL = "glytoucan_id"
DEFAULT_UNK_TOKEN = "[UNK]"

# -------------------------
# Dataset Class for GNN
# -------------------------
class GlycanGraphDataset(Dataset):
    """Dataset for loading glycan graphs from JSON files"""
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

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


def load_wurcs_from_csv(path: Path, wurcs_col: str, id_col: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(path, dtype=str)
    if wurcs_col not in df.columns:
        raise ValueError(f"[{path}] column '{wurcs_col}' not found. Available columns: {df.columns.tolist()}")
    
    if id_col not in df.columns:
        ids = [f"glycan_{i:06d}" for i in range(len(df))]
    else:
        ids = df[id_col].fillna("").replace("", np.nan)
        ids = ids.fillna(pd.Series([f"glycan_{i:06d}" for i in range(len(df))]))
        ids = ids.astype(str).tolist()

    wurcs_list = df[wurcs_col].astype(str).tolist()
    return list(zip(ids, wurcs_list))


def load_wurcs_from_txt(path: Path) -> List[Tuple[str, str]]:
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
    out_dir: Path,
    on_unknown: str = "skip",
) -> Tuple[List[str], Path]:
    """Convert WURCS strings to GNN graph JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parser = WURCSParser()

    manifest: List[str] = []
    graph_ids: List[str] = []

    for i, (gid, wurcs) in enumerate(tqdm(pairs, desc="WURCS→Graph(JSON)")):
        # Skip empty or invalid WURCS strings
        if not wurcs or not isinstance(wurcs, str):
            print(f"[Warn] skip {gid}: empty WURCS string")
            continue
        
        # Skip WURCS that doesn't start with "WURCS="
        if not wurcs.startswith("WURCS="):
            print(f"[Warn] skip {gid}: invalid WURCS format (does not start with 'WURCS=')")
            continue
            
        try:
            graph_data = parser.parse(wurcs)
        except Exception as e:
            if on_unknown == "skip":
                print(f"[Warn] skip {gid}: {e}")
                continue
            elif on_unknown == "unk":
                print(f"[Warn] (fallback=skip) {gid}: {e}")
                continue
            else:
                raise

        out_path = out_dir / f"{gid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        manifest.append(str(out_path))
        graph_ids.append(gid)

    manifest_path = out_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest), encoding="utf-8")
    return graph_ids, manifest_path


def load_vocab(path):
    """Load vocabulary from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab_sizes = {}
    token_to_idx = {}
    for feat_name, mapping in raw.items():
        vocab_sizes[feat_name] = len(mapping)
        token_to_idx[feat_name] = mapping
    return vocab_sizes, token_to_idx


def get_mask_token_idx(token_to_idx, feat_name):
    """Get MASK token index from vocab, fallback to vocab_size if not found"""
    if feat_name in token_to_idx:
        vocab_mapping = token_to_idx[feat_name]
        if "<MASK>" in vocab_mapping:
            return vocab_mapping["<MASK>"]
        return len(vocab_mapping)
    return -1


def build_hetero_graph(sample, token_to_idx):
    """Build heterogeneous graph from parsed WURCS data"""
    graph = sample
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
    big_node_maskable = []
    ring_bond_candidates_list = []
    rb_maskable_list = []
    double_bond_candidates_list = []
    db_maskable_list = []

    for bn in big_nodes:
        carbon_count_val = bn["carbon_count_type"]
        anomeric_pos_val = bn.get("anomeric_position_type", "")
        anomeric_config_val = bn.get("anomeric_config_type", "")
        
        carbon_count_list.append(token_to_idx["carbon_count_type"].get(
            carbon_count_val, 3))  # 默认索引 3（特殊embedding表示）
        anomeric_pos_list.append(token_to_idx["anomeric_position_type"].get(
            anomeric_pos_val, 0))  # 默认索引 0
        anomeric_config_list.append(token_to_idx["anomeric_config_type"].get(
            anomeric_config_val, 0))  # 默认索引 0
        
        has_question = "?" in carbon_count_val or "?" in anomeric_config_val
        has_x = "x" == anomeric_config_val.lower()
        big_node_maskable.append(1 if not (has_question or has_x) else 0)

        rbs = bn.get("ring_bonds", [])
        candidates = []
        rb_maskable = []
        for rb in rbs:
            rb_type = rb["ring_bond_type"]
            type_idx = token_to_idx["ring_bond_type"].get(
                rb_type, 0)  # 默认索引 0
            prob = rb.get("probability", 1.0)
            candidates.append((type_idx, prob))
            rb_has_question = "?" in rb_type
            rb_is_prob_one = (prob == 1.0)
            rb_maskable.append(1 if (not rb_has_question and rb_is_prob_one) else 0)
        if not candidates:
            candidates.append((0, 1.0))  # 默认索引 0
            rb_maskable.append(0)
        ring_bond_candidates_list.append(candidates)
        rb_maskable_list.append(rb_maskable)

        dbs = bn.get("double_bonds", ["none"])
        db_candidates = []
        db_maskable = []
        for db in dbs:
            type_idx = token_to_idx["double_bonds"].get(db, 0)  # 默认索引 0
            db_candidates.append(type_idx)
            db_maskable.append(1)
        if not db_candidates:
            db_candidates.append(0)  # 默认索引 0
            db_maskable.append(1)
        double_bond_candidates_list.append(db_candidates)
        db_maskable_list.append(db_maskable)

    big_x = torch.stack([
        torch.tensor(carbon_count_list, dtype=torch.long),
        torch.tensor(anomeric_pos_list, dtype=torch.long),
        torch.tensor(anomeric_config_list, dtype=torch.long),
    ], dim=1)

    rb_type_indices = []
    rb_probs = []
    rb_node_ptr = [0]
    for candidates in ring_bond_candidates_list:
        for type_idx, prob in candidates:
            rb_type_indices.append(type_idx)
            rb_probs.append(prob)
        rb_node_ptr.append(rb_node_ptr[-1] + len(candidates))

    rb_type_indices_t = torch.tensor(rb_type_indices, dtype=torch.long)
    rb_probs_t = torch.tensor(rb_probs, dtype=torch.float)
    rb_node_ptr_t = torch.tensor(rb_node_ptr, dtype=torch.long)

    db_type_indices = []
    db_node_ptr = [0]
    for db_candidates in double_bond_candidates_list:
        for type_idx in db_candidates:
            db_type_indices.append(type_idx)
        db_node_ptr.append(db_node_ptr[-1] + len(db_candidates))

    db_type_indices_t = torch.tensor(db_type_indices, dtype=torch.long)
    db_node_ptr_t = torch.tensor(db_node_ptr, dtype=torch.long)

    substituent_list = []
    small_node_maskable = []
    stereo_determined_small_node_ids = set()
    
    for se in small_edges:
        st = se.get("stereo_type", "?")
        if st in ['d', 'm'] or st in {'E', 'F', 'e', 'f', 'Z', 'z', 'n', 'N'}:
            stereo_determined_small_node_ids.add(se["small_node_id"])

    for sn_idx, sn in enumerate(small_nodes):
        substituent_val = sn["substituent"]
        substituent_list.append(token_to_idx["substituent"].get(
            substituent_val, 13))  # 默认索引 13
        base_maskable = 1 if "?" not in substituent_val else 0
        if sn_idx in stereo_determined_small_node_ids:
            base_maskable = 1
        small_node_maskable.append(base_maskable)
    
    small_x = torch.tensor(substituent_list, dtype=torch.long).unsqueeze(-1)
    small_node_maskable_t = torch.tensor(small_node_maskable, dtype=torch.float).unsqueeze(-1)

    big_edge_index = [[], []]
    big_edge_type_list = []
    big_edge_prob_list = []
    big_edge_maskable = []
    
    for be in big_edges:
        big_edge_index[0].append(be["source_big_node_id"])
        big_edge_index[1].append(be["target_big_node_id"])
        big_edge_type_val = be["big_edge_type"]
        big_edge_type_list.append(token_to_idx["big_edge_type"].get(
            big_edge_type_val, 34))  # 默认索引 34
        prob = be.get("probability", 1.0)
        big_edge_prob_list.append(prob)
        has_question = "?" in big_edge_type_val
        is_prob_one = (prob == 1.0)
        big_edge_maskable.append(1 if (not has_question and is_prob_one) else 0)

    if big_edge_index[0]:
        big_edge_index_t = torch.tensor(big_edge_index, dtype=torch.long)
        big_edge_attr_t = torch.tensor(big_edge_type_list, dtype=torch.long).unsqueeze(-1)
        big_edge_prob_t = torch.tensor(big_edge_prob_list, dtype=torch.float).unsqueeze(-1)
    else:
        big_edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        big_edge_attr_t = torch.zeros((0, 1), dtype=torch.long)
        big_edge_prob_t = torch.zeros((0, 1), dtype=torch.float)

    small_edge_index = [[], []]
    stereo_type_list = []
    carbon_pos_list = []
    small_edge_prob_list = []
    small_edge_maskable = []
    
    for se in small_edges:
        small_edge_index[0].append(se["big_node_id"])
        small_edge_index[1].append(se["small_node_id"])
        stereo_type_val = se["stereo_type"]
        stereo_type_list.append(token_to_idx["stereo_type"].get(
            stereo_type_val, 14))  # 默认索引 14
        carbon_pos_list.append(token_to_idx["carbon_position_type"].get(
            se["carbon_position_type"], 0))  # 默认索引 0
        prob = se.get("probability", 1.0)
        small_edge_prob_list.append(prob)
        has_question = "?" in stereo_type_val
        is_x = stereo_type_val == "x"
        is_prob_one = (prob == 1.0)
        small_edge_maskable.append(1 if (not has_question and not is_x and is_prob_one) else 0)

    if small_edge_index[0]:
        small_edge_index_t = torch.tensor(small_edge_index, dtype=torch.long)
        small_edge_attr_t = torch.stack([
            torch.tensor(stereo_type_list, dtype=torch.long),
            torch.tensor(carbon_pos_list, dtype=torch.long),
        ], dim=1)
        small_edge_prob_t = torch.tensor(small_edge_prob_list, dtype=torch.float).unsqueeze(-1)
        small_edge_maskable_t = torch.tensor(small_edge_maskable, dtype=torch.float).unsqueeze(-1)
    else:
        small_edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        small_edge_attr_t = torch.zeros((0, 2), dtype=torch.long)
        small_edge_prob_t = torch.zeros((0, 1), dtype=torch.float)
        small_edge_maskable_t = torch.zeros((0, 1), dtype=torch.float)

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

    hetero_data = HeteroData()

    hetero_data["big_node"].x = big_x
    hetero_data["big_node"].maskable = big_node_maskable_t
    hetero_data["small_node"].x = small_x
    hetero_data["small_node"].maskable = small_node_maskable_t

    hetero_data["big_node", "big_edge", "big_node"].edge_index = big_edge_index_t
    hetero_data["big_node", "big_edge", "big_node"].edge_attr = big_edge_attr_t
    hetero_data["big_node", "big_edge", "big_node"].edge_prob = big_edge_prob_t
    hetero_data["big_node", "big_edge", "big_node"].maskable = big_edge_maskable_t

    hetero_data["big_node", "small_edge", "small_node"].edge_index = small_edge_index_t
    hetero_data["big_node", "small_edge", "small_node"].edge_attr = small_edge_attr_t
    hetero_data["big_node", "small_edge", "small_node"].edge_prob = small_edge_prob_t
    hetero_data["big_node", "small_edge", "small_node"].maskable = small_edge_maskable_t

    if small_edge_index_t.size(1) > 0:
        rev_small_edge_index_t = torch.stack([small_edge_index_t[1], small_edge_index_t[0]], dim=0)
    else:
        rev_small_edge_index_t = torch.zeros((2, 0), dtype=torch.long)
    
    hetero_data["small_node", "rev_small_edge", "big_node"].edge_index = rev_small_edge_index_t
    hetero_data["small_node", "rev_small_edge", "big_node"].edge_attr = small_edge_attr_t
    hetero_data["small_node", "rev_small_edge", "big_node"].edge_prob = small_edge_prob_t
    hetero_data["small_node", "rev_small_edge", "big_node"].maskable = small_edge_maskable_t

    hetero_data.global_feat = torch.tensor([
        float(metadata.get("template_count", 0)),
        float(metadata.get("sugar_count", num_big)),
        float(metadata.get("true_linkage_count", 0)),
    ], dtype=torch.float)

    hetero_data.rb_type_indices = rb_type_indices_t
    hetero_data.rb_probs = rb_probs_t
    hetero_data.rb_node_ptr = rb_node_ptr_t
    hetero_data.rb_maskable = rb_maskable_t

    hetero_data.db_type_indices = db_type_indices_t
    hetero_data.db_node_ptr = db_node_ptr_t
    hetero_data.db_maskable = db_maskable_t

    hetero_data.big_node_batch = torch.zeros(num_big, dtype=torch.long)
    hetero_data.small_node_batch = torch.zeros(num_small, dtype=torch.long)

    return hetero_data


class GlycanDatasetForEmbedding(Dataset):
    """Dataset for loading glycan graphs and preparing for embedding extraction"""
    def __init__(self, raw_data_list, token_to_idx):
        self.raw_data_list = raw_data_list
        self.token_to_idx = token_to_idx
        self.graphs = []
        self.glycan_ids = []
        self.wurcs_list = []

        print("Building graphs...")
        for i, (gid, wurcs, data) in enumerate(raw_data_list):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(raw_data_list)} samples")
            try:
                g = build_hetero_graph(data, token_to_idx)
                if g is None:
                    raise ValueError("build_hetero_graph returned None")
                self.graphs.append(g)
                self.glycan_ids.append(gid)
                self.wurcs_list.append(wurcs)
            except Exception as e:
                print(f"[Warn] Failed to process {gid}: {e}")

        print(f"Done. Total: {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.glycan_ids[idx], self.wurcs_list[idx]


def collate_fn_for_embedding(batch):
    """Collate function for embedding extraction"""
    graphs, ids, wurcs = zip(*batch)

    batched_graph = HeteroData()

    for node_type in ["big_node", "small_node"]:
        xs = []
        batches = []
        maskables = []
        for i, g in enumerate(graphs):
            xs.append(g[node_type].x)
            batches.append(torch.full((g[node_type].x.size(0),), i, dtype=torch.long))
            maskables.append(g[node_type].maskable)
        batched_graph[node_type].x = torch.cat(xs, dim=0)
        batched_graph[node_type].batch = torch.cat(batches, dim=0)
        batched_graph[node_type].maskable = torch.cat(maskables, dim=0)

    for edge_type_key in [("big_node", "big_edge", "big_node"),
                          ("big_node", "small_edge", "small_node"),
                          ("small_node", "rev_small_edge", "big_node")]:
        if edge_type_key in graphs[0].edge_types:
            edge_indices = []
            edge_attrs = []
            edge_probs = []
            edge_maskables = []

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
                src_offset += g[src_type].x.size(0)
                dst_offset += g[dst_type].x.size(0)

            if edge_indices:
                batched_graph[edge_type_key].edge_index = torch.cat(edge_indices, dim=1)
                batched_graph[edge_type_key].edge_attr = torch.cat(edge_attrs, dim=0)
                batched_graph[edge_type_key].edge_prob = torch.cat(edge_probs, dim=0)
                batched_graph[edge_type_key].maskable = torch.cat(edge_maskables, dim=0)
            else:
                batched_graph[edge_type_key].edge_index = torch.zeros((2, 0), dtype=torch.long)
                batched_graph[edge_type_key].edge_attr = torch.zeros((0, graphs[0][edge_type_key].edge_attr.size(1)), dtype=torch.long)
                batched_graph[edge_type_key].edge_prob = torch.zeros((0, 1), dtype=torch.float)
                batched_graph[edge_type_key].maskable = torch.zeros((0, 1), dtype=torch.float)
            continue
            
        batched_graph[edge_type_key].edge_index = torch.zeros((2, 0), dtype=torch.long)
        batched_graph[edge_type_key].edge_attr = torch.zeros((0, 1), dtype=torch.long)
        batched_graph[edge_type_key].edge_prob = torch.zeros((0, 1), dtype=torch.float)
        batched_graph[edge_type_key].maskable = torch.zeros((0, 1), dtype=torch.float)

    # Merge ring_bond data
    rb_type_indices_list = []
    rb_probs_list = []
    rb_maskable_list = []
    rb_node_ptr_parts = []
    rb_candidate_offset = 0
    for i, g in enumerate(graphs):
        rb_type_indices_list.append(g.rb_type_indices)
        rb_probs_list.append(g.rb_probs)
        rb_maskable_list.append(g.rb_maskable)
        ptr = g.rb_node_ptr.clone()
        if i == 0:
            rb_node_ptr_parts.append(ptr + rb_candidate_offset)
        else:
            rb_node_ptr_parts.append(ptr[1:] + rb_candidate_offset)
        rb_candidate_offset += ptr[-1].item()

    batched_graph.rb_type_indices = torch.cat(rb_type_indices_list, dim=0)
    batched_graph.rb_probs = torch.cat(rb_probs_list, dim=0)
    batched_graph.rb_maskable = torch.cat(rb_maskable_list, dim=0)
    batched_graph.rb_node_ptr = torch.cat(rb_node_ptr_parts, dim=0)

    # Merge double_bond data
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

    return batched_graph, global_feats, list(ids), list(wurcs)


# -------------------------
# Model Loading
# -------------------------
def load_pretrained_model(checkpoint_path, device, vocab_sizes):
    """Load pretrained GNN model from checkpoint"""
    print(f"[Load] {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location=device)
    
    # Extract model configuration from checkpoint
    hidden_dim = state.get("hidden_dim", 256)
    num_layers = state.get("num_layers", 4)
    num_heads = state.get("num_heads", 4)
    dropout = state.get("dropout", 0.2)
    
    # Import model class from masktrain_windows_313 (supports edge existence prediction heads)
    from masktrain_windows_313 import HeteroGNN
    
    model = HeteroGNN(
        vocab_sizes=vocab_sizes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        mlm_probability=0.15
    )

    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


# -------------------------
# Embedding Extraction
# -------------------------
def extract_graph_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Extract graph embeddings using the pretrained model"""
    all_vecs = []
    all_ids = []
    all_wurcs = []
    
    with torch.no_grad():
        for batched_graph, global_feat, ids, wurcs in tqdm(loader, desc="Extract embeddings"):
            batched_graph = batched_graph.to(device)
            global_feat = global_feat.to(device)
            
            x_dict = {nt: batched_graph[nt].x for nt in batched_graph.node_types}
            edge_index_dict = {et: batched_graph[et].edge_index for et in batched_graph.edge_types}
            edge_attr_dict = {et: batched_graph[et].edge_attr for et in batched_graph.edge_types}
            edge_prob_dict = {et: batched_graph[et].edge_prob for et in batched_graph.edge_types}
            batch_dict = {nt: batched_graph[nt].batch for nt in batched_graph.node_types}
            
            features = model(
                x_dict, edge_index_dict, edge_attr_dict, edge_prob_dict,
                batch_dict, global_feat, batched_graph, return_mlm_logits=False
            )
            
            reps = features.detach().cpu().numpy()
            all_vecs.append(reps)
            all_ids.extend(ids)
            all_wurcs.extend(wurcs)

    if not all_vecs:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.concatenate(all_vecs, axis=0), all_ids, all_wurcs


# -------------------------
# Output Saving
# -------------------------
def save_outputs(
    out_dir: Path,
    graph_ids: List[str],
    embeddings: np.ndarray,
    wurcs_list: List[str] = None,
    prefix: str = "embeddings",
):
    """Save embeddings to output files"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NPY
    npy_path = out_dir / f"{prefix}.npy"
    np.save(npy_path, embeddings)
    
    # Save CSV with embeddings
    cols = [f"dim_{i}" for i in range(embeddings.shape[1])] if embeddings.size else []
    df = pd.DataFrame(embeddings, index=graph_ids, columns=cols)
    df.index.name = "glycan_id"
    
    # Add WURCS column if provided
    if wurcs_list:
        df["wurcs"] = wurcs_list
    
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path)

    # Save IDs
    (out_dir / f"{prefix}_ids.txt").write_text("\n".join(graph_ids), encoding="utf-8")

    print(f"[Save] NPY: {npy_path}")
    print(f"[Save] CSV: {csv_path}")
    print(f"[Save] IDs: {out_dir / f'{prefix}_ids.txt'}")


# -------------------------
# Per-file Processing
# -------------------------
def process_single_csv(
    input_csv_path: Path,
    output_csv_path: Path,
    model: torch.nn.Module,
    vocab_sizes: Dict,
    token_to_idx: Dict,
    device: torch.device,
    args,
    tmp_graph_dir: Path,
):
    """Process a single CSV file: add embeddings column while preserving original structure"""
    print(f"\n[Processing] {input_csv_path.name}")
    
    # Read original CSV
    df_original = pd.read_csv(input_csv_path, dtype=str)
    original_columns = df_original.columns.tolist()
    
    # Get WURCS and ID columns
    wurcs_col = args.wurcs_col
    id_col = args.id_col
    
    if wurcs_col not in df_original.columns:
        print(f"[Warn] '{wurcs_col}' column not found in {input_csv_path.name}, skipping.")
        return
    
    # Generate IDs if not present
    if id_col not in df_original.columns:
        ids = [f"glycan_{i:06d}" for i in range(len(df_original))]
    else:
        ids = df_original[id_col].fillna("").replace("", np.nan)
        ids = ids.fillna(pd.Series([f"glycan_{i:06d}" for i in range(len(df_original))]))
        ids = ids.astype(str).tolist()
    
    wurcs_list = df_original[wurcs_col].astype(str).tolist()
    pairs = list(zip(ids, wurcs_list))
    
    if not pairs:
        print(f"[Warn] No valid rows in {input_csv_path.name}, skipping.")
        return
    
    # WURCS -> graph JSON
    graph_ids, _ = encode_to_graph_json(
        pairs,
        out_dir=tmp_graph_dir,
        on_unknown=args.on_unknown,
    )
    
    if not graph_ids:
        print(f"[Warn] No graphs generated for {input_csv_path.name}, skipping.")
        return
    
    # Load parsed graph data
    raw_data_list = []
    for gid in graph_ids:
        graph_path = tmp_graph_dir / f"{gid}.json"
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        wurcs = next((w for (id_, w) in pairs if id_ == gid), "")
        raw_data_list.append((gid, wurcs, data))
    
    # Create dataset and loader
    dataset = GlycanDatasetForEmbedding(raw_data_list, token_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_for_embedding,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Extract embeddings
    embeddings, final_ids, final_wurcs = extract_graph_embeddings(model, loader, device)
    
    # Build mapping: id -> embedding
    id_to_embedding = {gid: emb for gid, emb in zip(final_ids, embeddings)}
    
    # Create new DataFrame with single embedding column
    df_output = df_original.copy()
    
    # Add single embedding column with bracket format
    df_output["embedding"] = None
    
    # Fill embeddings for each row
    for idx, gid in enumerate(ids):
        if gid in id_to_embedding:
            emb = id_to_embedding[gid]
            # Format as bracketed string: "[0.1, 0.2, 0.3, ...]"
            emb_str = "[" + ", ".join([str(v) for v in emb]) + "]"
            df_output.at[idx, "embedding"] = emb_str
    
    # Save output CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv_path, index=False)
    print(f"[Save] {output_csv_path} ({len(final_ids)} embeddings)")
    
    # Cleanup temp graphs for this file
    for gid in graph_ids:
        graph_path = tmp_graph_dir / f"{gid}.json"
        if graph_path.exists():
            graph_path.unlink()


# -------------------------
# Main
# -------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ap = argparse.ArgumentParser(description="Get glycan embeddings from WURCS strings.")
    
    ap.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "split_files"),
                    help="Directory containing input .csv and/or .txt with WURCS.")
    ap.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "split_files_processed_wurcs"),
                    help="Directory to write embeddings and logs.")
    ap.add_argument("--vocab_path", type=str, default=os.path.join(script_dir, "embedding_vocab.json"),
                    help="Path to embedding vocab JSON.")
    ap.add_argument("--checkpoint_path", type=str, default=os.path.join(script_dir, "checkpoints", "best_model_mlm.pt"),
                    help="Path to pretrained model checkpoint.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Number of workers for DataLoader. Use 0 on Windows to avoid multiprocessing issues.")
    ap.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Default: auto")
    ap.add_argument("--wurcs_col", type=str, default=DEFAULT_WURCS_COL,
                    help="CSV column name for WURCS.")
    ap.add_argument("--id_col", type=str, default=DEFAULT_ID_COL,
                    help="CSV column name for glycan ID.")
    ap.add_argument("--keep_graphs", action="store_true",
                    help="Keep temporary graph JSON files.")
    ap.add_argument("--on_unknown", type=str, default="skip", choices=["skip", "error"],
                    help="How to handle invalid WURCS strings.")
    
    args = ap.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tmp_graph_dir = output_dir / "tmp_graphs"

    if not input_dir.exists():
        raise FileNotFoundError(f"--input_dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_graph_dir.mkdir(parents=True, exist_ok=True)

    # Gather input CSV files only (preserve CSV structure)
    files = list(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No .csv found under {input_dir}")

    # Load vocab
    print(f"\nLoading vocabulary from {args.vocab_path}")
    vocab_sizes, token_to_idx = load_vocab(args.vocab_path)
    print(f"Vocab sizes: {vocab_sizes}")

    # Load model (shared across all files)
    print(f"\nLoading pretrained model from {args.checkpoint_path}")
    model = load_pretrained_model(args.checkpoint_path, device, vocab_sizes)

    # Process each CSV file separately
    for input_csv in sorted(files):
        output_csv = output_dir / input_csv.name
        process_single_csv(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            model=model,
            vocab_sizes=vocab_sizes,
            token_to_idx=token_to_idx,
            device=device,
            args=args,
            tmp_graph_dir=tmp_graph_dir,
        )

    # Cleanup temp graphs directory
    if not args.keep_graphs:
        shutil.rmtree(tmp_graph_dir, ignore_errors=True)
        print(f"[Clean] Removed temporary graphs: {tmp_graph_dir}")
    else:
        print(f"[Keep] Temporary graphs kept at: {tmp_graph_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
