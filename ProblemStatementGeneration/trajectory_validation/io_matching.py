# io_schema_matcher.py
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import traceback
from trajectory_validation.utils import Embedder, api_to_embeddings, safe_cosine
# =============================================
# CONFIGURATION
# =============================================
TYPE_BONUS = 0.25
NAME_WEIGHT = 0.6
MIN_SCORE_THRESHOLD = 0.6  # Only consider pairs with score >= 60%

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


    pair_scores.sort(key=lambda x: x["score"], reverse=True)

    visited_out = set()
    visited_in = set()
    matched_pairs = []

    for rec in pair_scores:
        if rec["pred_idx"] in visited_out or rec["succ_idx"] in visited_in:
            continue
        matched_pairs.append(rec)
        visited_out.add(rec["pred_idx"])
        visited_in.add(rec["succ_idx"])

    user_inputs = [
        succ_inputs[j][0] for j in range(len(succ_inputs))
        if j not in visited_in
    ]

    return matched_pairs, user_inputs


def validate_api_pair(
    pred_apis: List[Dict],
    succ_apis: List[Dict],
    embedder: Embedder,
    api_name_map: Dict[str, str]
) -> Dict[str, Any]:
    pred_outputs = []
    for api in pred_apis:
        _, outs = api_to_embeddings(api, embedder)
        api_name = api_name_map.get(api['name'], api['name'])
        for path, n_emb, typ, d_emb in outs:
            pred_outputs.append((f"{api_name}.{path}", n_emb, typ, d_emb))

    succ_inputs = []
    for api in succ_apis:
        ins, _ = api_to_embeddings(api, embedder)
        api_name = api_name_map.get(api['name'], api['name'])
        for path, n_emb, typ, d_emb in ins:
            succ_inputs.append((f"{api_name}.{path}", n_emb, typ, d_emb))

    matched_pairs, user_inputs_full = max_bipartite_matching(pred_outputs, succ_inputs)

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


def validate_trajectory(
    trajectory: List[Tuple[List[Any], float]],
    all_apis: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    print("[validate_trajectory] START", flush=True)
    print(f"[validate_trajectory] trajectory length: {len(trajectory)}", flush=True)
    print(f"[validate_trajectory] all_apis length: {len(all_apis)}", flush=True)

    api_lookup = {api['name']: api for api in all_apis}
    api_name_map = {api['name']: api['name'] for api in all_apis}
    print(f"[validate_trajectory] Built api_lookup with {len(api_lookup)} entries", flush=True)

    def clean_name(x) -> str:
        s = str(x)
        m = re.match(r"^np\.str_\(['\"](.*)['\"]\)$", s)
        cleaned = m.group(1) if m else s.strip()
        print(f"[clean_name] raw: {s} -> cleaned: {cleaned}", flush=True)
        return cleaned

    try:
        print("[validate_trajectory] Instantiating Embedder...", flush=True)
        embedder = Embedder()
        print("[validate_trajectory] Embedder instantiated", flush=True)
    except Exception as e:
        print("[validate_trajectory] ERROR instantiating Embedder:", e, flush=True)
        traceback.print_exc()
        raise

    results = []
    if len(trajectory) < 2:
        print("[validate_trajectory] Trajectory has fewer than 2 elements, returning empty results", flush=True)
        return results

    total_pairs = len(trajectory) - 1
    print(f"[validate_trajectory] Validating {total_pairs} trajectory pairs", flush=True)

    for i in tqdm(range(total_pairs), desc="Validating trajectory pairs"):
        try:
            print("\n" + "="*60, flush=True)
            print(f"[loop] i={i}", flush=True)
            pred_group, pred_score = trajectory[i]
            succ_group, succ_score = trajectory[i + 1]

            print(f"[loop] predecessor raw group (index {i}): {pred_group} (score={pred_score})", flush=True)
            print(f"[loop] successor raw group (index {i+1}): {succ_group} (score={succ_score})", flush=True)

            pred_names = [clean_name(n) for n in pred_group]
            succ_names = [clean_name(n) for n in succ_group]
            print(f"[loop] predecessor cleaned names: {pred_names}", flush=True)
            print(f"[loop] successor cleaned names: {succ_names}", flush=True)

            pred_apis = [api_lookup[n] for n in pred_names if n in api_lookup]
            succ_apis = [api_lookup[n] for n in succ_names if n in api_lookup]
            missing_pred = [n for n in pred_names if n not in api_lookup]
            missing_succ = [n for n in succ_names if n not in api_lookup]

            print(f"[loop] matched pred_apis count: {len(pred_apis)}; missing_pred: {missing_pred}", flush=True)
            print(f"[loop] matched succ_apis count: {len(succ_apis)}; missing_succ: {missing_succ}", flush=True)

            if not pred_apis or not succ_apis:
                print("[loop] One of pred_apis or succ_apis is empty — preparing empty-match result", flush=True)
                try:
                    succ_params = []
                    for api in succ_apis:
                        p = api.get("params", {})
                        print(f"[loop] succ api '{api.get('name')}' params (raw): {p}", flush=True)
                        if isinstance(p, dict):
                            print(f"[loop] succ api '{api.get('name')}' params keys: {list(p.keys())}", flush=True)
                        succ_params.extend(p if isinstance(p, list) else list(p))
                except Exception as e:
                    print("[loop] ERROR while extracting succ_apis params:", e, flush=True)
                    traceback.print_exc()

                results.append({
                    "predecessor": pred_names,
                    "successor": succ_names,
                    "matches": [],
                    "user_inputs": [p for api in succ_apis for p in api.get("params", {})]
                })
                print(f"[loop] Appended empty-match result for pair i={i}", flush=True)
                continue

            print("[loop] Calling validate_api_pair(...) with pred_apis and succ_apis", flush=True)
            try:
                result = validate_api_pair(pred_apis, succ_apis, embedder, api_name_map)
                print(f"[loop] validate_api_pair returned: {result}", flush=True)
            except Exception as e:
                print("[loop] ERROR in validate_api_pair:", e, flush=True)
                traceback.print_exc()
                result = {"matches": [], "error": str(e)}

            result.update({"predecessor": pred_names, "successor": succ_names})
            print(f"[loop] Final result for pair i={i}: {result}", flush=True)
            results.append(result)

        except Exception as e:
            print(f"[loop] Unexpected ERROR processing pair i={i}:", e, flush=True)
            traceback.print_exc()
            results.append({
                "predecessor": pred_group if 'pred_group' in locals() else None,
                "successor": succ_group if 'succ_group' in locals() else None,
                "matches": [],
                "error": f"Exception at pair {i}: {e}"
            })

    print("="*60, flush=True)
    print(f"[validate_trajectory] DONE — produced {len(results)} results", flush=True)
    return results