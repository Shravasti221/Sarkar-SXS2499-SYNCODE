import networkx as nx
import numpy as np
from tqdm import tqdm
import random
from typing import List, Tuple, Dict, Any
from graph_sampling.params import RANDOM_SEED

def select_diverse_trajectories_jaccard(
    G: nx.DiGraph,
    api_meta: Dict[str, Dict],
    emb_map: Dict[str, np.ndarray],
    candidate_trajectories: List[Tuple[List[str], float]],
    target_count: int = 100,
    jaccard_threshold: float = 0.35,
    keep_higher_entropy_prob: float = 0.9,
    n_gram: int = 3,
    rng=np.random.default_rng(RANDOM_SEED)
) -> List[Tuple[List[str], float]]:
    """
    From a large pool of (traj, score), keep exactly `target_count` trajectories
    such that no two have n-gram Jaccard similarity > `jaccard_threshold`.
    """
    if len(candidate_trajectories) == 0:
        return []

    # Build API → ID map
    api_to_id = {node: i for i, node in enumerate(sorted(G.nodes()))}

    def traj_to_ngrams(traj: List[str]) -> np.ndarray:
        if len(traj) < n_gram:
            return np.array([], dtype=np.int64)
        ids = [api_to_id.get(n, -1) for n in traj]
        grams = [hash(tuple(ids[i:i+n_gram])) for i in range(len(ids) - n_gram + 1)]
        return np.array(sorted(set(grams)), dtype=np.int64)

    def jaccard(sig1: np.ndarray, sig2: np.ndarray) -> float:
        if len(sig1) == 0 or len(sig2) == 0:
            return 0.0
        inter = np.intersect1d(sig1, sig2, assume_unique=True)
        union = np.union1d(sig1, sig2)
        return len(inter) / len(union)

    # Precompute signatures and scores
    sigs = [traj_to_ngrams(traj) for traj, _ in candidate_trajectories]
    scores = [score for _, score in candidate_trajectories]

    # Shuffle indices
    indices = list(range(len(candidate_trajectories)))
    rng.shuffle(indices)

    selected_idx = []
    selected_sigs = []

    pbar = tqdm(total=target_count, desc="Selecting diverse (Jaccard)")

    for idx in indices:
        if len(selected_idx) >= target_count:
            break

        traj_sig = sigs[idx]
        reject = False

        # --- Check against all selected ---
        for sel_pos, sel_sig in enumerate(selected_sigs):
            if jaccard(traj_sig, sel_sig) > jaccard_threshold:
                # Conflict found
                sel_idx = selected_idx[sel_pos]
                current_better = scores[idx] > scores[sel_idx]

                if current_better:
                    if rng.random() < keep_higher_entropy_prob:
                        # Replace old
                        selected_idx[sel_pos] = idx
                        selected_sigs[sel_pos] = traj_sig
                    else:
                        reject = True
                else:
                    if rng.random() < keep_higher_entropy_prob:
                        reject = True
                    else:
                        # Replace old
                        selected_idx[sel_pos] = idx
                        selected_sigs[sel_pos] = traj_sig
                        
                break  # only one conflict needed

        if not reject:
            selected_idx.append(idx)
            selected_sigs.append(traj_sig)
            pbar.update(1)

    pbar.close()
    print(f"[diversity] Kept {len(selected_idx)} diverse trajectories (Jaccard ≤ {jaccard_threshold})")

    return [(candidate_trajectories[i][0], candidate_trajectories[i][1]) for i in selected_idx]
    
