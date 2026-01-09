from typing import Any, Dict
import json
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage

from utils.helpers import print_message
from utils.llm import llm
from utils.safe_invoke_llm import safe_invoke_llm
from utils.pydantic_objects import responseFormat, write_pydantic_object
from utils.json_format import JsonFormat
# -----------------------------
# Orchestrator (Junior Assistant)
# -----------------------------
class Orchestrator:
    def __init__(self, name="orchestrator", experts:Dict=None):
        self.name = name
        if experts is None:
            with open("experts.json") as f:
                self.experts = json.load(f)
        else:
            self.experts = experts
        allowed_routes = tuple(self.experts.keys()) + ("api_execution", "response")
        self.json_format = JsonFormat(
            pydantic_model=responseFormat,
            expected_format=json.dumps(responseFormat.model_json_schema(), indent=2) + "\nAllowed routes: " + " or ".join(allowed_routes),
            verbose=True
        )


    def build_orchestrator_prompt(self, state: Dict[str, Any]) -> str:
        experts_keys_bar = " or ".join(tuple(self.experts.keys()))  # for the route example
        experts = [str(e) + ": " + t["description"] for e, t in self.experts.items()]
        most_recent_chat = []
        if len(state.chat_history) >0:
          most_recent_chat = state.chat_history[-1].content
        BASE_PROMPT = ""
        if most_recent_chat:
            if isinstance(state.chat_history[-1], (AIMessage, ToolMessage)):
                BASE_PROMPT = f"""
You are the Orchestrator Agent.
You have just received a message from an Expert.
Your job: Decide whether to
A) Forward the expert’s result to the user (route = "response")
B) Route the conversation to another expert (route = "<EXPERT_NAME>")

LATEST MESSAGE (for reference): {most_recent_chat}

--- DECISION RULES (STRICT JSON output only) ---
Output exactly one JSON object and nothing else. No extra keys, no explanation, no prose.

1) If the expert reply fully answers or concludes the topic, output EXACTLY:
{{{{ "route": "response", "task": None, "response": None }}}}

2) If the expert requests clarification or more input ask the user for clarification (return to user):
{{{{ "route": "response", "task": None, "response": None }}}}

--- ALLOWED EXPERT NAMES ---
Allowed Expert Names: {experts_keys_bar}
""".strip()
     
            elif isinstance(state.chat_history[-1], (HumanMessage)):
                # removed ability to give generic advice on unavailable tools like google calendar notifications, projector set up, etc.
                # as quality was low and often led to meaningless conversations.
                BASE_PROMPT = f"""
            You are the Orchestrator Agent.
            Your job is to:
            1. Interpret the user’s message and the full conversation history.
            2. Output a single JSON object and nothing else. Be BRIEF
            3. IF there is either {experts_keys_bar} has a match greater than 50% with the latest chat context ({most_recent_chat}) route to the correct Expert, output EXACTLY:
            {{
                "route": "<EXPERT_NAME>",
                "task": None,
                "response": None
            }}
            Allowed Expert Names: {experts_keys_bar}

            Experts available:
            {"\n".join(experts)}
            Json formatted response is: 
            """
        return BASE_PROMPT


    def node_fn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with LLM output (always return dict)."""
        sys_prompt = self.build_orchestrator_prompt(state)
        # print("[Orchestrator IP] Chat history length:", len(state.chat_history))
        
        msg = safe_invoke_llm(llm,[SystemMessage(content =  sys_prompt)] + state.chat_history)
        print_message(state, self.name, msg.content)
        
        formatted_output = self.json_format.refine(msg.content)
        msg_text = formatted_output.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(msg_text)
        expert_obj = responseFormat(**parsed)
        state.caller = self.name
        if expert_obj.route == "user" or expert_obj.route == "response":
            state.next = "user"
            return state
        if expert_obj.route in self.experts:
            state.next = expert_obj.route
            return state
        

    def route(self, state: Dict[str, Any]) -> str:
        """
        Interpret the LLM output (msg_content) and mutate `state` appropriately.
        Returns a small dict with the next action, e.g. {"next": "api_execution"} or {"next": "<expert_name>"}.
        """
        write_pydantic_object(state, state.ts)
        input()
        return state.next
