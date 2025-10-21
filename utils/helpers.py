from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from typing import List
from pydantic_objects import EventState
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


