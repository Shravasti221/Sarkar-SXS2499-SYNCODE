import json, time
import os
from dotenv import load_dotenv

from typing import TypedDict, List, Dict, Optional, Optional, Literal, Any
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv()

expert_json = json.load(open("experts.json"))
# Create a Literal from expert_json keys plus the fixed ones
allowed_routes = tuple(expert_json.keys()) + ("api_execution", "response")
# RouteType = Literal[allowed_routes]

class ApiCall(BaseModel):
    name: str = Field(..., description="Name of the API to call")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional dictionary of parameters for the API call"
    )
    
class responseFormat(BaseModel):
    route: str = Field(..., description="Type of route: "+ "or ".join(allowed_routes))
    task: Optional[ApiCall] = Field(
        None, description="API call details (required if route=='api_execution')"
    )
    response: Optional[str] = Field(
        None, description="Response string (required if route=='response')"
    )
    @field_validator("route")
    def validate_route(cls, v):
        if v not in allowed_routes:
            raise ValueError(f"Invalid route '{v}'. Allowed routes: {allowed_routes}")
        return v

    @model_validator(mode="after")
    def check_fields_based_on_route(self):
        if self.route not in allowed_routes:
            raise ValueError(f"Invalid route '{self.route}'. Allowed routes: {allowed_routes}")
        if self.route == "api_execution" and not self.task:
            raise ValueError("task must be provided when route='api_execution'")
        if self.route == "response" and not self.response:
            raise ValueError("response must be provided when route='response'")
        return self

# -----------------------------
# State definition
# -----------------------------
class EventState(BaseModel):
    problem_created: Optional[str] = None
    # chat_history: List[Dict[str, str]] = Field(default_factory=list)
    chat_history: List[BaseMessage] = []
    api_task: Optional[Dict[str, Any]] = None
    api_result: Optional[dict] = None
    caller: Optional[str] = None
    next: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


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


# -----------------------------
# LLM Initialization
# -----------------------------
if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY not found")

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    # reasoning_effort="low",
     model_kwargs={
        "stream": False,
        "tool_choice": "none",  # Explicitly disable tool calls
        # Optionally enforce JSON output
        # "response_format": {"type": "json_object"}
    })


