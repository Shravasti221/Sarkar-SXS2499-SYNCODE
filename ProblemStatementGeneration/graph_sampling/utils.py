from typing import List, Tuple, Dict, Any
import pandas as pd
import re
import json


def split_at_capitals(word: str) -> str:
    """Split camelCase/PascalCase into spaced words."""
    if not word:
        return ""
    parts = re.findall(r'[A-Z][^A-Z]*', word)
    if not parts:
        # fallback: split on non-alpha
        return re.sub(r'[_\-]+', ' ', word)
    return " ".join(parts)

def find_csv_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Find likely source, target, weight columns in dataframe by heuristics.
    Returns (src_col, dst_col, weight_col) or raises ValueError.
    """
    cols = [c.strip().lower() for c in df.columns]
    # possible names
    src_candidates = [i for i, c in enumerate(cols) if any(k in c for k in ("n1", "source", "from", "api1", "src"))]
    dst_candidates = [i for i, c in enumerate(cols) if any(k in c for k in ("n2", "target", "to", "api2", "dst"))]
    w_candidates = [i for i, c in enumerate(cols) if any(k in c for k in ("weight", "weights", "score", "w"))]

    def pick(cands):
        return cands[0] if cands else None

    si = pick(src_candidates)
    di = pick(dst_candidates)
    wi = pick(w_candidates)

    if si is None or di is None or wi is None:
        # fallback: try common exact names
        mapping = {c.lower(): c for c in df.columns}
        for s in ("n1_name", "source", "from", "api1_name", "api1"):
            if s in mapping and si is None:
                si = df.columns.get_loc(mapping[s])
        for d in ("n2_name", "target", "to", "api2_name", "api2"):
            if d in mapping and di is None:
                di = df.columns.get_loc(mapping[d])
        for w in ("weight", "weights", "score"):
            if w in mapping and wi is None:
                wi = df.columns.get_loc(mapping[w])

    if si is None or di is None or wi is None:
        raise ValueError(f"Could not autodetect CSV columns. Found: {list(df.columns)}")

    return df.columns[si], df.columns[di], df.columns[wi]

# ----------------------------
# 1. Load & normalize experts JSON
# ----------------------------
def normalize_api_structure(data: dict) -> dict:
    sensitive_keys = {"url","auth_key","apikey","key","access_token","auth_token","authorization","x-api-key","auth_header"}
    for category, category_data in data.items():
        normalized = []
        for entry in category_data.get("apis", []):
            if isinstance(entry, dict) and all(k in entry for k in ("APIName","params","description","output")):
                name = entry["APIName"]
                params = {k:v for k,v in entry.get("params", {}).items() if k.lower() not in sensitive_keys}
                out = entry.get("output", {})
                if isinstance(out, str): out = {"type": out}
                normalized.append({"APIName": name, "params": params, "description": entry.get("description",""), "output": out})
            elif isinstance(entry, dict) and len(entry)==1 and isinstance(next(iter(entry.values())), dict):
                name = next(iter(entry.keys()))
                info = next(iter(entry.values()))
                params = {k:v for k,v in info.get("params", {}).items() if k.lower() not in sensitive_keys}
                out = info.get("output", {})
                if isinstance(out, str): out = {"type": out}
                normalized.append({"APIName": name, "params": params, "description": info.get("description",""), "output": out})
            else:
                # skip malformed
                print(f"[normalize] Skipped malformed API entry in '{category}': {entry}")
        category_data["apis"] = normalized
    return data

def load_experts(experts_json_path: str) -> Tuple[Dict[str, Dict[str,Any]], Dict[str, str]]:
    """
    Return:
      - api_meta: api_name -> dict(description, params, output, expert)
      - expert_map: api_name -> expert_name (duplicate)
    """
    with open(experts_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = normalize_api_structure(data)
    api_meta: Dict[str, Dict[str,Any]] = {}
    for expert_name, block in data.items():
        for api in block.get("apis", []):
            name = api["APIName"]
            if name in api_meta:
                # merge descriptions if duplicates appear (append)
                prev = api_meta[name]
                prev_desc = prev.get("description","")
                new_desc = api.get("description","")
                if new_desc and new_desc not in prev_desc:
                    prev["description"] = (prev_desc + " " + new_desc).strip()
                # don't overwrite params/outputs; keep first by default
            else:
                api_meta[name] = {
                    "description": api.get("description",""),
                    "params": api.get("params", {}),
                    "output": api.get("output", {}),
                    "expert": expert_name
                }
    return api_meta


def save_results_csv(results: List[Tuple[List[str], float]], api_meta: Dict[str, Dict[str,Any]], out_csv: str):
    rows = []
    for traj, score in results:
        experts = sorted({ api_meta.get(n, {}).get("expert","") for n in traj if api_meta.get(n, {}).get("expert","") })
        rows.append({
            "trajectory": " -> ".join(traj),
            "length": len(traj),
            "expert_count": len(experts),
            "expert_list": ", ".join(experts),
            "score": float(score)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[save] saved {len(df)} rows to {out_csv}")
