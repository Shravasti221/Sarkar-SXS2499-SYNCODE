import csv
import networkx as nx
import numpy as np
import json
import time
import random

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from graph_sampling.utils import load_experts


def safe_invoke_llm(llm, messages, max_retries=3, retry_delay=3, **kwargs):
    """
    Invokes the LLM with built-in retry on Groq rate limit responses.
    Handles both structured JSON errors and Python exceptions.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages, **kwargs)
            # Check if response looks like a JSON error blob
            try:
                data = json.loads(response.content)
                if isinstance(data, dict) and "error" in data:
                    err = data["error"]
                    if isinstance(err, dict) and err.get("code") == "rate_limit_exceeded":
                        print(f"[Attempt {attempt}] Rate limit hit — retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
            except Exception:
                # Not a JSON string, proceed as normal
                pass

            # Normal success path
            return response

        except Exception as e:
            # Some APIs raise directly instead of returning a JSON error
            msg = str(e)
            if "rate limit" in msg.lower() or "rate_limit_exceeded" in msg.lower():
                print(f"[Attempt {attempt}] Exception: Rate limit reached — retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                # Non-rate-limit exception — stop retrying
                raise

    raise RuntimeError(f"LLM failed after {max_retries} attempts due to repeated rate limit errors.")

def build_nx_graph_from_csv(path):
    """
    Reads a CSV of the form:
        n1_name, n2_name, weight
    and builds a weighted directed NetworkX graph.
    """
    G = nx.DiGraph()

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["n1_name"].strip()
            dst = row["n2_name"].strip()

            # If weight column exists
            weight = float(row.get("weight", 1.0))

            G.add_edge(src, dst, weight=weight)

    return G

def sample_nodes_by_eigen(G, k, replace=False):
    """
    Randomly samples k graph nodes using probabilities proportional
    to their eigenvector centrality.
    """
    # Compute eigenvector centrality (normalized by default)
    eig = nx.eigenvector_centrality(G, max_iter=500, weight='weight')

    nodes = list(eig.keys())
    weights = np.array([eig[n] for n in nodes], dtype=float)

    # Normalize to sum to 1
    probs = weights / weights.sum()

    # Random choice weighted by eigenvector centrality
    sampled = np.random.choice(
        nodes,
        size=k,
        replace=replace,
        p=probs
    )

    return list(sampled)

from typing_extensions import Literal, TypedDict, Annotated
import os
from collections import deque
# ------------------------------
# Trajectory generation
# ------------------------------

def generate_trajectory(G, start_node, p_stop=0.2, rejection_prob=0.9):
    trajectory = deque([start_node])
    selection_counts = {n: 0 for n in G.nodes}

    # Expand parents
    while np.random.rand() > p_stop:
        parents = list(G.predecessors(trajectory[0]))
        if not parents:
            break
        weights = np.array([G[p][trajectory[0]].get('weight', 1) + selection_counts[p] for p in parents], dtype=float)
        probs = weights / weights.sum()
        parent = np.random.choice(parents, p=probs)
        if parent in trajectory and np.random.rand() < rejection_prob:
            break
        trajectory.appendleft(parent)
        selection_counts[parent] += 1

    # Expand children
    while np.random.rand() > p_stop:
        children = list(G.successors(trajectory[-1]))
        if not children:
            break
        weights = np.array([G[trajectory[-1]][c].get('weight', 1) + selection_counts[c] for c in children], dtype=float)
        probs = weights / weights.sum()
        child = np.random.choice(children, p=probs)
        if child in trajectory and np.random.rand() < rejection_prob:
            break
        trajectory.append(child)
        selection_counts[child] += 1

    return list(trajectory)

def generate_trajectories_for_sampled_nodes(G, k, p_stop=0.2, rejection_prob=0.9):
    sampled_nodes = sample_nodes_by_eigen(G, k)
    trajectories = []
    for node in sampled_nodes:
        traj = generate_trajectory(G, node, p_stop=p_stop, rejection_prob=rejection_prob)
        trajectories.append([str(t) for t in traj])
    return trajectories

class Scenario(TypedDict):
    problem_statement: list[str]  # e.g., ["You are organizing a corporate event..."]
    steps: list[str]  # e.g., ["fetchEventDetails", "searchVenues", ...]
    hints: list[str]  # e.g., ["Consider the event date and location...", ...]
    steps_with_hints: dict[tuple, str]  # e.g., {(1, 'fetchEventDetails', 'searchVenues'): "hint"}

# ------------------------------
# API-scenario mapping
# ------------------------------
EXPERTS_JSON_PATH = "/content/experts_Event Management Company Simplified.json"
G = build_nx_graph_from_csv("/content/api_graph.csv")
trajectories = generate_trajectories_for_sampled_nodes(G, 10, p_stop=0.3, rejection_prob=0.95)

apis = load_experts(EXPERTS_JSON_PATH)
all_scenarios = []
for trajectory in trajectories:
    scenario_apis = [(api_name, apis[api_name]) for api_name in trajectory]
    if scenario_apis:
        scenario = Scenario(
            problem_statement=["Generated scenario from trajectory"],
            steps=scenario_apis,
            hints=[],
            steps_with_hints={(i, scenario_apis[i][0], scenario_apis[i+1][0]): ""
                              for i in range(len(scenario_apis)-1)}
        )
        scenario["steps_with_hints"][(0, "start_dialogue", scenario_apis[0][0])] = ""
        all_scenarios.append(scenario)


def reduce_list(left: list | None, right: list | dict | None) -> list:
    left = left or []
    
    # Handle cases where a node returns a single dict instead of list
    if isinstance(right, dict):
        right = [right]
    elif right is None:
        right = []
    elif isinstance(right, list):
        pass
    else:
        right = [right]
    return left + right


class State(TypedDict):
    # scenarios: Annotated[list[Scenario], 'custom_reducer=reduce_list']
    scenarios: Annotated[list[Scenario], reduce_list]
    # vote: Annotated[list[int], 'custom_reducer=reduce_list']
    route: int

# ------------------------------
# Prompt functions
# ------------------------------

def stage1_prompt(scenario: Scenario) -> tuple[str, str]:
  SYSTEM_PROMPT = """You are the creative director of a scriptwriting company.
  Your role is to take a list of API calls and create a coherent scenario that uses all of them in order."""
  # Always return an array of strings."""

  #     HINT_GENERATION_PROMPT = f"""You are a creative director for a scriptwriting company. Your task is to create a **complete scenario** that uses a sequence of API calls.
  # You are given the following scenario information:
  # * **Steps (API calls to use in order):**
  # {scenario['steps']}

  # * **Step-to-hint mapping (for guidance between steps):**
  # {scenario['steps_with_hints']}

  # **Your task:**
  # 1. Generate a coherent scenario or storyline that **uses all of the API calls in the order specified** by `{scenario['steps']}`.
  # 2. For each transition between steps as defined in `{list(scenario['steps_with_hints'].keys())}`, generate a **helpful hint**.
  # 3. Return your answer **only as an array of hints** in this format:

  # ["hint0 (start_dialogue)", "hint1", "hint2", "hint3", ...]

  # **Rules:**
  # * Do not include extra text, explanations, or formatting—only the array of hints.
  # * Each hint should correspond exactly to the transitions between consecutive steps.
  # * Hints should be creative, clear, and actionable within the scenario context.

  # hints = """

  HINT_GENERATION_PROMPT = f"""You are a creative director for a scriptwriting company. Your task is to create a complete scenario that uses a sequence of API calls.
  You are given the following scenario information:

  * **Step-to-hint mapping (for guidance between steps):**
  { "start_dialogue" + "->" + "->".join([api_name[0] for api_name in scenario['steps']])}

  API Call definitions : {json.dumps(scenario['steps'], indent = 2)}

  **Your task:**
  1. Generate a coherent scenario or storyline that **uses all of the API calls in the order specified** by `{"start_dialogue" + "->" + "->".join([api_name[0] for api_name in scenario['steps']])}`.
  2. For each transition between steps as defined in {list(scenario['steps_with_hints'].keys())}, generate a **helpful hint** that adds the correct `(apiname)` wherever it should be called in the problem statement.
  3. Return a detailed problem statment that will use all api calls by adding the correct acpi call in the scenario statement.

  **Rules:**
  * Each hint should correspond exactly to the transitions between consecutive steps.
  * Hints to which api_name to be used should be added as `(api_name)` within the scenario context.
  * all apis should be used in given order.
  * Do NOT use any formatting except `(api_name)`

  Scenario: """

  return SYSTEM_PROMPT, HINT_GENERATION_PROMPT

# def eval_stage_prompt(scenario: Scenario) -> tuple[str, str]:
#     SYSTEM_PROMPT = """You are a pipeline flow evaluator.
# Given a pair of API calls and the hint between them, evaluate if moving from API1 to API2 is logically consistent according to the hint.
# Return only 1 if valid, 0 if invalid."""

#     USER_PROMPT = f"""You are given the following scenario:
# * Steps: {scenario['steps']}
# * Hints: {scenario['hints']}

# For each consecutive pair of APIs (api1, api2) with the corresponding hint:
# - Output 1 if moving from api1 to api2 logically follows according to the hint.
# - Output 0 if it does not.

# Return your answer as an array of 1s and 0s matching the order of step transitions, e.g.:

# [1, 0, 1]
# """

#     return SYSTEM_PROMPT, USER_PROMPT


# ------------------------------
# Scene generator
# ------------------------------

def fetch_scene_node(state: State):
    return state

def scene_generator(state: State, model_name: str) -> State:
    llm = ChatOpenAI(
        model=model_name,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,
        stream=False,
    )

    sys_prompt, user_prompt = stage1_prompt(state['scenarios'][0])
    print("SYS PROMPT: ", sys_prompt)
    print("USER_PROMPT: ", user_prompt)
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = safe_invoke_llm(llm, messages)
    return Scenario(problem_statement= response.content)

def scene_generator_1(state: State):
    return {"scenarios": scene_generator(state, "openai/gpt-oss-20b")}

def scene_generator_2(state: State):
    return {"scenarios": scene_generator(state, "llama-3.3-70b-versatile")}

def scene_generator_3(state: State):
    return {"scenarios": scene_generator(state, "openai/gpt-oss-120b")}

#     if not isinstance(parse_response, list) or not all(x in [0, 1] for x in parse_response):
#         raise ValueError("Invalid response: Expected list of 0s and 1s.")

#     # Vote: 1 if all transitions are valid, else 0
#     state['vote'] = 1 if all(parse_response) else 0
#     return state

# def eval_trajectory_node_1(state: State):
#     return eval_trajectory_node(state, "openai/gpt-oss-20b")

# def eval_trajectory_node_2(state: State):
#     return eval_trajectory_node(state, "llama-3.3-70b-versatile")

# def eval_trajectory_node_3(state: State):
#     return eval_trajectory_node(state, "openai/gpt-oss-120b")


# ------------------------------
# Build graph
# ------------------------------

def generate_problem_statement(scenario: Scenario) -> Scenario:
    builder = StateGraph(State)
    builder.add_node("fetch_scene_node", fetch_scene_node)
    builder.add_node("scene_generator_1", scene_generator_1)
    builder.add_node("scene_generator_2", scene_generator_2)
    builder.add_node("scene_generator_3", scene_generator_3)

    builder.add_edge(START, "fetch_scene_node")
    builder.add_edge("fetch_scene_node", "scene_generator_1")
    builder.add_edge("fetch_scene_node", "scene_generator_2")
    builder.add_edge("fetch_scene_node", "scene_generator_3")

    builder.add_edge("scene_generator_1", END)
    builder.add_edge("scene_generator_2", END)
    builder.add_edge("scene_generator_3", END)

    graph = builder.compile()

    ret_val = graph.invoke(State(scenarios=[scenario]))
    scenario = ret_val["scenarios"][1] 
    scenario["problem_statement"] = [ret_val["scenarios"][random.choice([2, 3, 4])]["problem_statement"]]
    return scenario