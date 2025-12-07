from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
import re
import uuid
from typing import List
from utils.pydantic_objects import EventState
# -----------------------------
# Helper function for logging
# -----------------------------
def print_message(state: EventState, role: str, content: str):
    line = f"[{role}] {content}"
    print(line)
    
def chat_history_to_string(chat_history: List[BaseMessage]) -> str:
    """
    Converts a list of LangChain messages into a readable string transcript,
    showing expertise for AI messages and toolname for Tool messages.
    """
    lines = []
    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            role = "SYSTEM"
        elif isinstance(msg, HumanMessage):
            role = "USER"
        elif isinstance(msg, AIMessage):
            role = "ASSISTANT"
        elif isinstance(msg, ToolMessage):
            role = "TOOL"
        else:
            role = msg.__class__.__name__.replace("Message", "").upper()

        # Add expertise for AI messages
        if isinstance(msg, AIMessage):
            expertise = msg.additional_kwargs.get("expertise")
            if expertise:
                role = f"{role} ({expertise})"

        # Add toolname for Tool messages
        if isinstance(msg, ToolMessage):
            toolname = msg.additional_kwargs.get("toolname")
            if toolname:
                role = f"{role} [{toolname}]"

        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def extract_and_order_steps(problem_data: dict, problem_index: int = 0):
    """
    Extract sentences for API calls, fill steps_with_hints,
    return sorted list, and attach UUIDs to each function
    as tuple pairs (function, uuid). Does NOT inject uuid into text.
    """

    # ---- Extract & clean text ----
    text = problem_data["problem_statement"][problem_index]
    cleaned = re.sub(r"\s+", " ", text)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    # ---- Build list of all API names ----
    api_list = [step_name for step_name, _ in problem_data["steps"]]
    api_list.append("start_dialogue")

    # ---- Assign a stable UUID for each function once ----
    uuid_map = {api: str(uuid.uuid4()) for api in api_list}

    # ---- Map API → sentences ----
    api_to_sentence = {}
    for api in api_list:
        pattern = re.compile(rf"\({api}\)", re.IGNORECASE)
        api_to_sentence[api] = [
            s.strip()
            for s in sentences
            if pattern.search(s)
        ]

    # ---- Build unsorted results ----
    results = []
    for key in problem_data["steps_with_hints"]:
        index, prev_api, curr_api = key

        prev_tuple = (prev_api, uuid_map.get(prev_api))
        curr_tuple = (curr_api, uuid_map.get(curr_api))

        extracted = api_to_sentence.get(curr_api, [])
        hint_sentence = " ".join(extracted) if extracted else ""

        results.append((index, prev_tuple, curr_tuple, hint_sentence))

    # ---- Custom sorting: start_dialogue previous always first ----
    def sort_key(item):
        index, prev_tuple, curr_tuple, _ = item
        prev_api, _ = prev_tuple
        if prev_api == "start_dialogue":
            return (-1, 0)
        return (index, 1)

    results.sort(key=sort_key)
    return results