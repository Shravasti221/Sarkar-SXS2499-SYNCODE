import json
from pprint import pformat
from typing import Any
from langchain_core.messages import BaseMessage
import json
from utils.pydantic_objects import EventState

def _serialize_value(v: Any):
    """Best-effort serialization for readable output."""
    if isinstance(v, BaseMessage):
        return {
            "type": v.__class__.__name__,
            "content": v.content
        }
    if hasattr(v, "model_dump"):
        return v.model_dump()
    return v


def pretty_print_event_state(
    state: EventState,
    *,
    max_width: int = 100,
    show_empty: bool = False
):
    """
    Pretty-print EventState in a human-readable, debug-friendly format.
    """

    data = state.model_dump()

    lines = []
    lines.append("=" * max_width)
    lines.append(f"🧠 EventState @ ts={state.ts}")
    lines.append("=" * max_width)

    for key in [
        "problem_created",
        "problem_statement",
        "caller",
        "next",
        "api_task",
        "api_result",
        "chat_history",
    ]:
        value = data.get(key)

        if not show_empty and not value:
            continue

        lines.append(f"\n▶ {key}")

        if key == "chat_history":
            if not value:
                lines.append("  (empty)")
            else:
                for i, msg in enumerate(state.chat_history):
                    msg_data = _serialize_value(msg)
                    lines.append(f"  [{i}] {msg_data['type']}:")
                    lines.append(
                        "    " + pformat(
                            msg_data["content"],
                            width=max_width - 8,
                            compact=True
                        )
                    )
        else:
            lines.append(
                pformat(
                    _serialize_value(value),
                    width=max_width,
                    compact=True
                )
            )

    lines.append("\n" + "=" * max_width)

    print("\n".join(lines))


def pretty_print_trajectory_with_hints(problem_trajectory):
    print("=" * 100)
    print("FULL PROBLEM TRAJECTORY WITH HINTS PER PROBLEM STATEMENT")
    print("=" * 100)
    print()

    # Extract common parts
    problem_statements = problem_trajectory['problem_statement']
    hints_per_statement = problem_trajectory['hints']          # list of lists
    steps = problem_trajectory['steps']
    # steps_with_hints = problem_trajectory['steps_with_hints']

    num_statements = len(problem_statements)
    if num_statements != len(hints_per_statement):
        print("   Problem Statement:")
        print("   " + "=" * 50)
        print("\n".join("   " + line for line in problem_statements[0].strip().split("\n")))
        print()

        print("   Associated Hints:")
        print("   " + "-" * 30)
        statement_hints = hints_per_statement
        # print(statement_hints)
        for hint_idx, (step_idx, from_node, to_node, hint_text) in enumerate(statement_hints, 1):
            print(f"   Hint {hint_idx}: {from_node} → {to_node}")
            print("       " + "\n       ".join(hint_text.strip().split("\n")))
            print()
        print("\n" + "=" * 80)
        print("\n")

        # 2. API Steps (shared across all variants)
        print("2. API STEPS SEQUENCE (shared)")
        print("-" * 60)
        print(f"Total steps: {len(steps)}\n")
        for idx, (api_name, details) in enumerate(steps, 1):
            print(f"{idx}. {api_name}")
            print(f"    Expert     : {details['expert']}")
            print(f"    Description: {details['description']}")
            print(f"    Params     : {json.dumps(details['params'], indent=8)[8:]}")
            print(f"    Output     : {json.dumps(details['output'], indent=8)[8:]}")
            print()
        print()
        return
        
    # assert num_statements == len(hints_per_statement), "Mismatch between statements and hints!"

    # 1. Print each Problem Statement with its corresponding hints
    print(f"1. PROBLEM STATEMENTS + DEDICATED HINTS ({num_statements} variants)")
    print("-" * 80)
    for i in range(num_statements):
        print(f"\nVARIANT {i+1} / {num_statements}")
        print("   Problem Statement:")
        print("   " + "=" * 50)
        print("\n".join("   " + line for line in problem_statements[i].strip().split("\n")))
        print()

        print("   Associated Hints:")
        print("   " + "-" * 30)
        statement_hints = hints_per_statement[i]
        for hint_idx, (step_idx, from_node, to_node, hint_text) in enumerate(statement_hints, 1):
            print(f"   Hint {hint_idx}: {from_node} → {to_node}")
            print("       " + "\n       ".join(hint_text.strip().split("\n")))
            print()
        print("\n" + "=" * 80)
    print("\n")

    # 2. API Steps (shared across all variants)
    print("2. API STEPS SEQUENCE (shared)")
    print("-" * 60)
    print(f"Total steps: {len(steps)}\n")
    for idx, (api_name, details) in enumerate(steps, 1):
        print(f"{idx}. {api_name}")
        print(f"    Expert     : {details['expert']}")
        print(f"    Description: {details['description']}")
        print(f"    Params     : {json.dumps(details['params'], indent=8)[8:]}")
        print(f"    Output     : {json.dumps(details['output'], indent=8)[8:]}")
        print()
    print()

    # 3. Steps-with-Hints Mapping (structural view)
    # print("3. STEPS_WITH_HINTS MAPPING (transition keys)")
    # print("-" * 60)
    # print("These are the transition pairs that hints refer to (currently values are empty):\n")
    # for key in steps_with_hints.keys():
    #     if len(key) == 3:
    #         idx, src, dst = key
    #         print(f"  ({idx}) {src} → {dst}")
    #     else:
    #         print(f"  {key}")
    print(f"\nTotal unique transitions: {len(steps_with_hints)}")
    print()

    # 4. Quick Summary
    print("4. QUICK SUMMARY")
    print("-" * 60)
    api_names = [step[0] for step in steps]
    print(f"Trajectory length : {len(api_names)} APIs")
    print(f"API chain         : {' → '.join(api_names)}")
    print(f"Experts involved  : {set(step[1]['expert'] for step in steps)}")
    print(f"Problem variants  : {num_statements}")
    print()
