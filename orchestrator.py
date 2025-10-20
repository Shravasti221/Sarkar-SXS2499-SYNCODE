from typing import Any, Dict
import json
import uuid
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage

from helpers import print_message, llm, safe_invoke_llm
from helpers import responseFormat
from formatter import Formatter

import difflib
from typing import List, Set

def get_repeated_experts(chat_history: List, threshold: float = 0.85, window: int = 4) -> Set[str]:
    """
    Detects experts who have sent algorithmically similar messages repeatedly.

    Args:
        chat_history (List): List of AIMessage objects with .content and .response_metadata.
        threshold (float): Similarity threshold for near-duplicate detection.
        window (int): Number of recent expert messages to check.

    Returns:
        Set[str]: Names of experts who have repeated themselves.
    """

    repeated_experts = set()

    expert_msgs = [
        msg for msg in reversed(chat_history)
        if hasattr(msg, "response_metadata")
        and msg.response_metadata.get("type") == "expert"
    ][:window]

    if len(expert_msgs) < 2:
        return repeated_experts

    for i in range(len(expert_msgs)):
        for j in range(i + 1, len(expert_msgs)):
            exp_i = expert_msgs[i].response_metadata.get("name")
            exp_j = expert_msgs[j].response_metadata.get("name")

            if exp_i != exp_j:
                continue

            text_i = expert_msgs[i].content.strip().lower()
            text_j = expert_msgs[j].content.strip().lower()

            ratio = difflib.SequenceMatcher(None, text_i, text_j).ratio()

            if ratio >= threshold:
                repeated_experts.add(exp_i)

    return repeated_experts



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
        self.formatter = Formatter(
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
        # Detect repeated experts in recent history
        repeated_experts = get_repeated_experts(state.chat_history)
        repetition_note = ""

        if repeated_experts:
            repeated_list = ", ".join(repeated_experts)
            repetition_note = f"""
            ---REPETITION NOTE ---
        IMPORTANT CONTEXT:
        The following expert(s) — {repeated_list} — have been giving very similar responses multiple times.
        Do NOT call them again unless you have new information or context.
        Explicitly decide whether to:
        - Route to a different expert with fresh reasoning, OR
        - Return the message to the user with a clarified or summarized response.
        """
        
          
          
        """Build the system prompt for the Orchestrator LLM."""
        BASE_PROMPT = ""
        if most_recent_chat:
            if isinstance(most_recent_chat, (AIMessage, ToolMessage)):
                BASE_PROMPT = f"""
You are the Orchestrator Agent.
You have just received a message from an Expert.
Your job: Decide whether to
A) Forward the expert’s result to the user (route = "response"), possibly adding short context or a clarifying question, OR
B) Route the conversation to another expert (route = "<EXPERT_NAME>") if the expert’s reply indicates the task is incomplete or requires different expertise.

LATEST MESSAGE (for reference): {most_recent_chat}

--- CONTEXT RULES (use all history) ---
- Treat the full conversation history as available context.
- Detect whether the latest expert reply **completes**, **deflects**, **requests clarification**, or **recommends another domain**.

--- DECISION RULES (STRICT JSON output only) ---
Your assistant MUST output exactly one JSON object and nothing else. No extra keys, no explanation, no prose.

1) If the expert reply fully answers or concludes the topic, output EXACTLY:
{{{{ "route": "response", "task": null, "response": "<summarized or contextualized version of the expert’s message for the user>" }}}}

2) If the expert reply clearly states inability, lack of tools, or suggests a different domain, route to an appropriate expert:
{{{{ "route": "<EXPERT_NAME>", "task": null, "response": null }}}}

3) If the expert requests clarification or more input ask the user for clarification (return to user):
{{{{ "route": "response", "task": null, "response": "<expert clarifying question to user; include context and why it's needed>" }}}}

--- ALLOWED EXPERT NAMES ---
Allowed Expert Names: {experts_keys_bar}

--- THINKING / BEHAVIORAL INSTRUCTIONS ---
- First, check whether the most recent message came from the same expert previously used for this topic.
- If {repetition_note.strip() and 'repetition_note is present' or 'no repetition_note present'}: account for repetition as instructed above.
- If the expert asks for clarification, prefer asking the **user** when the requested information is user-specific (credentials, preferences, consent, sample rows). Prefer routing to another expert/tool if the missing info can be derived automatically (e.g., data-cleaning agent).
- If routing to an expert, include one-line rationale in your head but DO NOT output it — only emit the JSON.
- Always follow the JSON templates exactly; no extra fields allowed.

Begin internal reasoning now, then emit EXACTLY one JSON object that conforms to the templates above. No prose, no markdown, no extra characters.
""".strip()
     
            elif isinstance(most_recent_chat, (HumanMessage)):
                BASE_PROMPT = f"""
            You are the Orchestrator Agent.
            Your job is to:
            1. Interpret the user’s message and the full conversation history.
            2. Decide exactly one of three actions:
            - Return a direct response (the Orchestrator answers).
            - Route to an Expert (ask an expert to act or answer).

            Rules for output format (STRICT JSON only — no prose, no extra fields, no markdown):
            1. Your assistant output MUST be a single JSON object and nothing else.
            2. IF there is no appropriate expert agent to call and you have the knowledge to respond to the chat, output EXACTLY:
            {{
                "route": "response",
                "task": null,
                "response": "<your textual answer>"
            }}
            3. ELSE IF there is an expert agent that has a match greater than 50% with the latest chat context ({most_recent_chat}) route to the Expert, output EXACTLY:
            {{
                "route": "<EXPERT_NAME>",
                "task": null,
                "response": null
            }}
            Allowed Expert Names: {experts_keys_bar}

            Experts available:
            {"\n".join(experts)}
            Begin reasoning _internally_, then emit EXACTLY one JSON object that conforms to the rules above.
            """
        return BASE_PROMPT


    def node_fn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with LLM output (always return dict)."""
        prompt = self.build_orchestrator_prompt(state)
        # print("\n========== ORCHESTRATOR DEBUG ==========")
        # print("Prompt to LLM:\n", prompt[:800], "...\n")
        print("[Orchestrator IP] Chat history length:", len(state.chat_history))
        
        prompt = self.build_orchestrator_prompt(state)
        msg = safe_invoke_llm(llm,[SystemMessage(content =  prompt)] + state.chat_history)
        print_message(state, self.name, msg.content)

        try:
            formatted_output = self.formatter.refine(msg.content)
            msg_text = formatted_output.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(msg_text)
        except Exception as e:
            # fallback if LLM fails to produce JSON
            print(f"Orchestrator JSON parse error: {e}")
            state.chat_history.append(ToolMessage(content=f"Error: {e} (JSON Parse Error, defaulting to JuniorAssistant) Raw Text: {msg.content}", tool_call_id= "json_parse_error" + str(uuid.uuid4()), response_metadata={"type": "error", "name": "json_parse_error"}))
            state.next = "orchestrator"
            return state
        try:
            expert_obj = responseFormat(**parsed)
        except Exception as e:
            # fallback
            print(f"Orchestrator JSON parse/validation error: {e}")
            state.chat_history.append(ToolMessage(content=f"Error: {e} (Validation Error, defaulting to JuniorAssistant) Raw Text: {msg.content}, Required Pydantic format: {responseFormat.model_json_schema()}", tool_call_id= "pydantic_parse_error" + str(uuid.uuid4()), response_metadata={"type": "error", "name": "pydantic_parse_error"}))
            state.next = "orchestrator"
            return state
        if expert_obj.route == "api_execution":
            # task must be present and valid
            if not expert_obj.task:
                state.next = "orchestrator"
                return state
            state.api_task = expert_obj.task.dict()
            state.caller = self.name
            state.next = "api_execution"
            return state
        if expert_obj.route == "response":
            response_text = expert_obj.response or ""
            state.chat_history.append(AIMessage(content=response_text, response_metadata={"type": "orchestrator", "name": self.name}))
            state.caller = self.name
            state.next = "user"
            return state
        if expert_obj.route in self.experts:
            response_text = expert_obj.response or "ROUTE TO " + expert_obj.route
            state.chat_history.append(AIMessage(content=response_text, response_metadata={"type": "orchestrator", "name": self.name}))
            state.caller = self.name
            state.next = expert_obj.route
            return state
        

    def route(self, state: Dict[str, Any]) -> str:
        """
        Interpret the LLM output (msg_content) and mutate `state` appropriately.
        Returns a small dict with the next action, e.g. {"next": "api_execution"} or {"next": "<expert_name>"}.
        """
        return state.next
