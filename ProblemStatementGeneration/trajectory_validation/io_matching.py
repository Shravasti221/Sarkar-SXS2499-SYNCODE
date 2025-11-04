# io_schema_matcher.py
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from utils import Embedder, api_to_embeddings, safe_cosine
# =============================================
# CONFIGURATION
# =============================================
TYPE_BONUS = 0.25
NAME_WEIGHT = 0.6
MIN_SCORE_THRESHOLD = 0.6  # Only consider pairs with score >= 60%



# =============================================
# GREEDY MATCHING (NEW METHOD)
# =============================================
def max_bipartite_matching(
    pred_outputs: List[Tuple[str, np.ndarray, str, np.ndarray]],
    succ_inputs: List[Tuple[str, np.ndarray, str, np.ndarray]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Greedy matching: all pairs → sort → pick if both unvisited.
    Returns:
        matched_pairs: list of full match records
        user_inputs:   list of unmatched input param names (full path)
    """
    if not pred_outputs or not succ_inputs:
        return [], [t[0] for t in succ_inputs]

    # 1. All pairwise scores
    pair_scores = []
    for i, (o_path, o_name_e, o_typ, o_desc_e) in enumerate(pred_outputs):
        for j, (i_path, i_name_e, i_typ, i_desc_e) in enumerate(succ_inputs):
            name_sim = safe_cosine(o_name_e, i_name_e)
            desc_sim = safe_cosine(o_desc_e, i_desc_e)
            type_match = (o_typ and i_typ and o_typ == i_typ)

            score = NAME_WEIGHT * name_sim + (1 - NAME_WEIGHT) * desc_sim
            if type_match:
                score += TYPE_BONUS

            if score >= MIN_SCORE_THRESHOLD:
                pair_scores.append({
                    "pred_idx": i,
                    "succ_idx": j,
                    "score": score,
                    "output_param": o_path,
                    "input_param": i_path,
                    "output_type": o_typ,
                    "input_type": i_typ,
                    "name_sim": name_sim,
                    "desc_sim": desc_sim,
                    "type_match": type_match
                })

    # 2. Sort descending
    pair_scores.sort(key=lambda x: x["score"], reverse=True)

    # 3. Greedy selection
    visited_out = set()
    visited_in = set()
    matched_pairs = []

    for rec in pair_scores:
        if rec["pred_idx"] in visited_out or rec["succ_idx"] in visited_in:
            continue
        matched_pairs.append(rec)
        visited_out.add(rec["pred_idx"])
        visited_in.add(rec["succ_idx"])

    # 4. Unmatched inputs
    user_inputs = [
        succ_inputs[j][0] for j in range(len(succ_inputs))
        if j not in visited_in
    ]

    return matched_pairs, user_inputs


# =============================================
# PAIRWISE VALIDATION
# =============================================
def validate_api_pair(
    pred_apis: List[Dict],
    succ_apis: List[Dict],
    embedder: Embedder,
    api_name_map: Dict[str, str]
) -> Dict[str, Any]:
    # Build outputs
    pred_outputs = []
    for api in pred_apis:
        _, outs = api_to_embeddings(api, embedder)
        api_name = api_name_map.get(api['name'], api['name'])
        for path, n_emb, typ, d_emb in outs:
            pred_outputs.append((f"{api_name}.{path}", n_emb, typ, d_emb))

    # Build inputs
    succ_inputs = []
    for api in succ_apis:
        ins, _ = api_to_embeddings(api, embedder)
        api_name = api_name_map.get(api['name'], api['name'])
        for path, n_emb, typ, d_emb in ins:
            succ_inputs.append((f"{api_name}.{path}", n_emb, typ, d_emb))

    # Greedy matching
    matched_pairs, user_inputs_full = max_bipartite_matching(pred_outputs, succ_inputs)

    # Strip API prefix
    def strip(path: str) -> str:
        return path.split('.', 1)[1] if '.' in path else path

    match_records = [
        {
            "output_param": strip(m["output_param"]),
            "output_type": m["output_type"],
            "input_param": strip(m["input_param"]),
            "input_type": m["input_type"],
            "score": round(m["score"], 6),
            "name_sim": round(m["name_sim"], 6),
            "desc_sim": round(m["desc_sim"], 6),
            "type_match": m["type_match"]
        }
        for m in matched_pairs
    ]

    user_inputs = [strip(p) for p in user_inputs_full]

    return {
        "matches": match_records,
        "user_inputs": sorted(set(user_inputs))
    }


# =============================================
# TRAJECTORY VALIDATOR
# =============================================
def validate_trajectory(
    trajectory: List[Tuple[List[Any], float]],
    all_apis: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    api_lookup = {api['name']: api for api in all_apis}
    api_name_map = {api['name']: api['name'] for api in all_apis}

    def clean_name(x) -> str:
        s = str(x)
        m = re.match(r"^np\.str_\(['\"](.*)['\"]\)$", s)
        return m.group(1) if m else s.strip()

    embedder = Embedder()
    results = []

    for i in tqdm(range(len(trajectory) - 1), desc="Validating trajectory pairs"):
        pred_group, _ = trajectory[i]
        succ_group, _ = trajectory[i + 1]

        pred_names = [clean_name(n) for n in pred_group]
        succ_names = [clean_name(n) for n in succ_group]

        pred_apis = [api_lookup[n] for n in pred_names if n in api_lookup]
        succ_apis = [api_lookup[n] for n in succ_names if n in api_lookup]

        if not pred_apis or not succ_apis:
            results.append({
                "predecessor": pred_names,
                "successor": succ_names,
                "matches": [],
                "user_inputs": [p for api in succ_apis for p in api.get("params", {})]
            })
            continue

        result = validate_api_pair(pred_apis, succ_apis, embedder, api_name_map)
        result.update({"predecessor": pred_names, "successor": succ_names})
        results.append(result)

    return results
