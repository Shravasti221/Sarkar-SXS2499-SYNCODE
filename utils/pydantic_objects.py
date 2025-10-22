import json
import os
from datetime import datetime
from typing import TypedDict, List, Dict, Optional, Optional, Literal, Any
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from langchain_core.messages import BaseMessage

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
    ts: str
    chat_history: List[BaseMessage] = []
    api_task: Optional[Dict[str, Any]] = None
    api_result: Optional[dict] = None
    caller: Optional[str] = None
    next: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

def create_timestamped_file(content: str="") -> str:
    """Write content to a file."""
    ts = ".\\converstions\\"+ datetime.now().isoformat(timespec='seconds')+".json"
    with open(ts, "w", encoding="utf-8") as f:
        f.write(content)
    return ts

def save_pydantic_object(obj: BaseModel, filepath: str):
    """Save a Pydantic model instance to a JSON file."""
    if not os.path.exists(str):
        raise FileNotFoundError(f"{str} does not exist")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj.model_dump(), f, indent=4)

def load_pydantic_object(model_class, filepath: str):
    """Load a Pydantic model instance from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model_class(**data)
