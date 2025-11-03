import json
import random
import re
import os
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import torch
import warnings
from helpers import split_at_capitals, find_csv_columns
from params import SENTENCE_MODEL, RANDOM_SEED, WEIGHT_DISSIM, WEIGHT_LENGTH, WEIGHT_EXPERT, LOOP_PENALTY_SCALE, MAX_LEN_NORM
warnings.filterwarnings("ignore")

# ----------------------------
# 2. Build Graph (string node ids)
# ----------------------------
def build_graph(api_meta: Dict[str, Dict[str,Any]], weights_csv_path: str) -> Tuple[nx.DiGraph, Dict[str, Dict[str,Any]]]:
    """
    Build directed graph with node ids = api_name (strings).
    Set node attribute 'meta' to api_meta[api_name].
    """
    if not os.path.exists(weights_csv_path):
        raise FileNotFoundError(weights_csv_path)

    df = pd.read_csv(weights_csv_path)
    src_col, dst_col, w_col = find_csv_columns(df)

    G = nx.DiGraph()
    # ensure all APIs present as nodes
    for name, meta in api_meta.items():
        G.add_node(name, meta=meta)

    missing_edges = 0
    for _, row in df.iterrows():
        src = str(row[src_col]).strip()
        dst = str(row[dst_col]).strip()
        try:
            w = float(row[w_col])
        except Exception:
            w = 0.0
        if src not in api_meta or dst not in api_meta:
            missing_edges += 1
            # we still add nodes if referenced but absent in JSON (optional)
            if src not in G:
                G.add_node(src, meta={"description":"","params":{},"output":{},"expert":""})
            if dst not in G:
                G.add_node(dst, meta={"description":"","params":{},"output":{},"expert":""})
        G.add_edge(src, dst, weight=w)
    # print summary
    print(f"[build_graph] nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, missing_refs={missing_edges}")
    return G, api_meta

# ----------------------------
# 3. Embeddings (precompute)
# ----------------------------
def compute_embeddings_for_meta(api_meta: Dict[str, Dict[str,Any]], model_name: str = SENTENCE_MODEL,
                                batch_size: int = 64, device: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Compute embedding vector for each API: use "Name. Description" as text.
    Returns mapping api_name -> 1D numpy array.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    api_names = list(api_meta.keys())
    texts = []
    for name in api_names:
        name_text = split_at_capitals(name)
        desc = api_meta[name].get("description","") or ""
        text = (name_text + ". " + desc).strip()
        texts.append(text)

    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    emb_map: Dict[str, np.ndarray] = {}
    for n, v in zip(api_names, embeddings):
        emb_map[n] = v.astype(np.float32)
    return emb_map

# ----------------------------
# 4. Sampling starting nodes (unique by api_name)
# ----------------------------
def sample_start_nodes_unique(G: nx.DiGraph, k: int, rng=random) -> List[str]:
    """
    Sample nodes weighted by out-degree. Return up to k UNIQUE node names.
    Nodes with zero out-degree are excluded (they cannot be starts if no outgoing).
    """
    nodes = list(G.nodes())
    out_degs = np.array([G.out_degree(n) for n in nodes], dtype=float)
    # exclude zero-out-degree nodes (they are sinks)
    mask = out_degs > 0
    cand_nodes = [n for n, m in zip(nodes, mask) if m]
    weights = out_degs[mask] if mask.any() else np.ones(len(nodes))
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    # If k >= len(cand_nodes), return all
    k_eff = min(k, len(cand_nodes))
    # use numpy choice without replacement via permutation
    chosen = list(np.random.choice(cand_nodes, size=k_eff, replace=False, p=probs))
    return chosen

# ----------------------------
# 5. Trajectory generation (uses string node ids)
# ----------------------------
def backward_to_root(G: nx.DiGraph, start: str, rng=random, max_steps: Optional[int]=None) -> List[str]:
    """
    Walk backwards by choosing a random predecessor at each step until node with no predecessors.
    Returns path root -> ... -> start.
    Uses visited set to avoid infinite cycles; stops if revisit or max_steps exceeded.
    """
    if max_steps is None:
        max_steps = max(1, G.number_of_nodes())
    visited = set()
    path = []
    cur = start
    while len(path) < max_steps:
        if cur not in G:
            break
        preds = list(G.predecessors(cur))
        if not preds:
            break
        nxt = rng.choice(preds)
        if nxt in visited:               # cycle → stop
            break
        visited.add(nxt)
        path.append(nxt)                 # predecessor
        cur = nxt
    path.reverse()
    return path

def forward_random_walk(G: nx.DiGraph, start: str, p_stop: float = 0.25, max_steps: int = 30, rng=random) -> List[str]:
    """
    Random walk forward from 'start' following outgoing edges.
    At each node, stop with probability p_stop BEFORE stepping (so can end at start).
    Edge selection weighted by positive edge weight (clamped).
    Returns list including start and subsequent nodes.
    """
    path = [start]
    cur = start
    steps = 0
    for _ in range(max_steps):
        if rng.random() < p_stop:
            break
        if cur not in G:
            break
        succs = list(G.successors(cur))
        if not succs:
            break
        raw_w = [max(0.0, G[cur][s].get('weight', 1.0)) for s in succs]
        total = sum(raw_w)
        if total <= 0:
            probs = [1.0 / len(succs)] * len(succs)
        else:
            probs = [w / total for w in raw_w]
        cur = rng.choices(succs, weights=probs, k=1)[0]
        path.append(cur)
    return path

def full_trajectory_from_start(G: nx.DiGraph, start: str, p_stop: float=0.25, max_back: Optional[int]=None, max_fwd:int=30, rng=random) -> List[str]:
    """
    Return a full trajectory: root -> ... -> start -> ... -> end
    - backward_to_root returns root->...->parent_of_start (no start)
    - forward_random_walk starts at start
    Combine without duplicate start
    """
    back = backward_to_root(G, start, rng=rng, max_steps=max_back)
    fwd = forward_random_walk(G, start, p_stop=p_stop, max_steps=max_fwd, rng=rng)
    # combine: back (may be []) + fwd
    if back and back[-1] == start:
        combined = back + fwd[1:]
    else:
        # ensure start present once
        combined = back + fwd
    return combined

# ----------------------------
# 6. Entropy function (uses precomputed embeddings)
# ----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assume normalized inputs; safe fallback
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (da * db))
# -------------------------------------------------
# 4. Entropy – global inverse centrality, proper penalty
# -------------------------------------------------
def _global_inv_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Pre-compute 1/(out_degree+ε) for every node once."""
    eps = 1e-8
    max_deg = max((d for _, d in G.out_degree()), default=1)
    inv = {}
    for n, d in G.out_degree():
        inv[n] = 1.0 / (d + eps)
    total = sum(inv.values())
    return {n: w / total for n, w in inv.items()}

# call once after graph construction:
INV_CENTRALITY = _global_inv_centrality(G)

def entropy_function(
    traj: List[str], G: nx.DiGraph, emb_map: Dict[str, np.ndarray],
    api_meta: Dict[str, Dict], *,
    weight_dissim=WEIGHT_DISSIM, weight_length=WEIGHT_LENGTH,
    weight_expert=WEIGHT_EXPERT, loop_penalty_scale=LOOP_PENALTY_SCALE,
    max_len_norm=MAX_LEN_NORM
) -> float:
    if len(traj) <= 1:
        return 0.0

    # ---- 1. dissimilarity (centrality weighted) ----
    inv_w = [INV_CENTRALITY.get(n, 0.0) for n in traj]
    total_w = sum(inv_w)
    if total_w == 0:
        inv_w = [1.0 / len(traj)] * len(traj)

    dissim_sum = 0.0
    pair_count = 0
    for i, n1 in enumerate(traj):
        v1 = emb_map.get(n1)
        if v1 is None:
            continue
        for j, n2 in enumerate(traj[i + 1 :], i + 1):
            v2 = emb_map.get(n2)
            if v2 is None:
                continue
            sim = np.dot(v1, v2)                     # already normalized
            dissim = 1.0 - sim
            w = (inv_w[i] + inv_w[j]) / 2.0
            dissim_sum += w * dissim
            pair_count += w
    dissim_score = dissim_sum / (pair_count + 1e-12)

    # ---- 2. length (log-scaled) ----
    length_score = np.log1p(len(traj)) / np.log1p(max_len_norm)

    # ---- 3. loop penalty (fraction of repeats) ----
    repeats = len(traj) - len(set(traj))
    repeat_frac = repeats / len(traj)
    loop_penalty = -loop_penalty_scale * repeat_frac   # ∈ [-1,0]

    # ---- 4. expert diversity (normalised) ----
    experts = {api_meta.get(n, {}).get("expert", "") for n in traj}
    experts.discard("")
    total_experts = len({m["expert"] for m in api_meta.values() if m.get("expert")})
    expert_norm = len(experts) / max(total_experts, 1)

    # ---- composite (all terms 0-1 except penalty) ----
    score = (weight_dissim * dissim_score +
             weight_length * length_score +
             weight_expert * expert_norm)
    score += loop_penalty
    return float(score)

# ----------------------------
# 7. Aggregate: sample starts, create multiple tra j per start, compute scores
# ----------------------------
def generate_and_score(
    G: nx.DiGraph,
    api_meta: Dict[str, Dict[str,Any]],
    emb_map: Dict[str, np.ndarray],
    n_start: int = 50,
    traj_per_node: int = 3,
    p_stop: float = 0.25,
    max_back: Optional[int] = None,
    max_fwd: int = 30,
    rng = random
) -> List[Tuple[List[str], float]]:
    starts = sample_start_nodes_unique(G, n_start)
    print(f"[generate] sampled {len(starts)} start nodes")
    results: List[Tuple[List[str], float]] = []
    for idx, s in enumerate(starts):
        # for each start, generate multiple trajectories
        for _ in range(traj_per_node):
            traj = full_trajectory_from_start(G, s, p_stop=p_stop, max_back=max_back, max_fwd=max_fwd, rng=rng)
            score = entropy_function(traj, G, emb_map, api_meta)
            results.append((traj, score))
    # sort descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results