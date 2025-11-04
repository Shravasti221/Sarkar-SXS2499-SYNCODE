"""
API Graph Trajectory Entropy Ranker

- Loads APIs from experts JSON and normalizes entries
- Builds a directed graph from a CSV of edge weights
- Precomputes text embeddings for each API (name + description)
- Samples starting nodes weighted by out-degree (unique)
- For each start node: find a backward root, then perform randomized forward walks
- Aggregate trajectories and compute an "entropy" score per trajectory using:
    1) centrality-weighted pairwise dissimilarity (from embeddings)
    2) trajectory length (longer -> higher)
    3) loop penalty (repeated nodes penalize)
    4) number of distinct experts (more -> higher)
- Output: CSV with ranked trajectories
"""
import random 
import numpy as np
import torch
from graph_sampling.sample_graph import build_graph, load_experts, compute_embeddings_for_meta
from graph_sampling.sample_graph import (
    generate_and_score,
    select_diverse_trajectories_jaccard,
    save_results_csv
)
from graph_sampling.params import RANDOM_SEED
from graph_sampling.utils import normalize_api_structure, load_experts
from trajectory_validation.io_matching import validate_trajectory

EXPERTS_JSON_PATH = "experts_Event Management Company.json"
def main(
    experts_json_path: str = "experts_Event Management Company.json",
    weights_csv_path: str = "eventManagement_apigraph_weights.csv",
    output_csv: str = "entropy_trajectories.csv",
    n_start: int = 50,
    traj_per_node: int = 3,
    p_stop: float = 0.25
):
    # reproducible
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("[main] loading expert metadata...")
    api_meta = load_experts(experts_json_path)

    print("[main] building graph...")
    G, api_meta = build_graph(api_meta, weights_csv_path)

    print("[main] computing embeddings (this may take a moment)...")
    emb_map = compute_embeddings_for_meta(api_meta)

    print("[main] generating & scoring trajectories...")
    raw_trajectories = generate_and_score(G, api_meta, emb_map, n_start=n_start, traj_per_node=traj_per_node, p_stop=p_stop)

    # # print top 10
    # print("\nTop 10 trajectories:")
    # for i, (traj, score) in enumerate(results[:10], start=1):
    #     experts = sorted({ api_meta.get(n, {}).get("expert","") for n in traj if api_meta.get(n, {}).get("expert","") })
    #     print(f"{i:2d}. score={score:.4f} len={len(traj):2d} experts={len(experts):d} path={ ' -> '.join(traj) }")

    print("[main] generating & scoring trajectories...")
    print(f"[main] generated {len(raw_trajectories)} raw trajectories. Now filtering for diversity...")
    results = select_diverse_trajectories_jaccard(
        G, api_meta, emb_map,
        candidate_trajectories=raw_trajectories,
        target_count=100,
        jaccard_threshold=0.35,
        keep_higher_entropy_prob=0.9,
        n_gram=3
    )

    # print top 10
    print("\nTop 10 DIVERSE high-entropy trajectories:")
    for i, (traj, score) in enumerate(results[:10], start=1):
        experts = sorted({api_meta.get(n, {}).get("expert","") for n in traj if api_meta.get(n, {}).get("expert","")})
        experts = [e for e in experts if e]
        print(f"{i:2d}. score={score:.4f} len={len(traj):2d} experts={len(experts):d}")
        print(f"     {' -> '.join(traj)}")
        if experts:
            print(f"     Experts: {', '.join(experts)}\n")

    save_results_csv(results, api_meta, output_csv)
    return results

if __name__ == "__main__":
    # Adjust paths/n_start/etc. as needed
    trajectories = main(
        experts_json_path="experts_Event Management Company.json",
        weights_csv_path="eventManagement_apigraph_weights.csv",
        output_csv="entropy_trajectories.csv",
        n_start=50,
        traj_per_node=3,
        p_stop=0.25
    )
    
    all_apis = []
    for key, value in normalize_api_structure(load_experts(EXPERTS_JSON_PATH)).items():
        all_apis+= [{"name": i["APIName"], "params":i["params"], "description": i["description"], "output":i["output"], "expert": key} for i in value["apis"]]
    results = validate_trajectory(trajectories[:4], all_apis)