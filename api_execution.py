import json
import uuid
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from utils.pydantic_objects import EventState
from utils.safe_invoke_llm import safe_invoke_llm
from utils.llm import llm
import re

class APIPipeline:
    def __init__(self, api_spec_path: str = "experts.json"):
        self.llm = llm
        self.api_specs = self._load_api_specs(api_spec_path)

    # ---------------------------------------------------------
    # Load all API definitions from JSON file
    # ---------------------------------------------------------
    def _load_api_specs(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"API spec file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)

        api_map = {}
        for agent_name, agent_data in data.items():
            for api_name, api_data in agent_data.get("apis", {}).items():
                api_map[api_name] = {
                    "params": api_data.get("params", {}),
                    "description": api_data.get("description", ""),
                    "agent": agent_name
                }
        print(f"[APIPipeline] Loaded {len(api_map)} APIs from {path.name}")
        return api_map

    
    def process(self, state:EventState):
        """Entry point: always start with clarification, then possibly execute API."""
        state = self.clarification_node(state)
        
        # If clarification passed validation, move to API execution
        if state.next == "api_execution":
            state = self.api_execution_node(state)
        return state

    # -------------------------------
    # 1. Clarification Node
    # -------------------------------
    def clarification_node(self, state:EventState)-> EventState:
        task = state.api_task
        caller = state.caller

        print(f"\n[ClarificationNode] Incoming task from: {caller}")
        print(f"[ClarificationNode] Raw task:\n{json.dumps(task, indent=2)}")

        system_prompt = f"""
        You are a strict API Parameter Validator & Clarifier.
        The expected API call structure is:
        {{
          "name": "<api_name>",
          "params": {{ "<param_name>": "<param_value>", ... }}
        }}

        Rules:
        1. Verify that all params required by the API are present and there are no null/placeholders.
           {{
             "status": "ok"
             "response": None
           }}
           or
           {{
             "status": "fail",
             "response": "Describe what’s wrong and why it cannot be fixed automatically. If clarification is needed, explain which parameter is missing."
           }}
           DONOT EXECUTE TOOL.
        """

        conversation_text = "\n".join([
            f"{m.type.upper()}: {m.content}" for m in state.chat_history[-10:]
        ])

        input_payload = {"api_task": task, "context": conversation_text}

        msg = safe_invoke_llm(self.llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(input_payload))
        ])

        print(f"[ClarificationNode] LLM Output:\n{msg.content}")

        try:
            match = re.search(r'{.*?}', msg.content, re.DOTALL)
            if match:
                clarification = match.group(0)
            else:
                clarification = "\"status\": \"fail\", \"issue\": \"No JSON object found in clarifier response. Current Feedback:" +msg.content +"}"
            result = json.loads(clarification)
        except Exception as e:
            print(f"[ClarificationNode] JSON Parse Error: {e}")
            result = {"status": "fail", "issue": "Non-JSON response from clarifier."}

        if result["status"] == "ok":
            print("[ClarificationNode] ✅ Parameters validated")
            state.next = "api_execution"
        else:
            issue = result.get("response", "Unknown validation issue.")
            print("[ClarificationNode] ❌ Irrecoverable issue:", issue)
            feedback = f"Cannot execute {task.get('name', 'unknown')}: {issue}"
            updated_history = state.chat_history + [AIMessage(content=feedback, response_metadata={"type": "expert", "name": "api_clarification"})]
            state.chat_history = updated_history
            state.next = caller
            state.caller = "api_execution"

        return state

    # -------------------------------
    # 2. API Execution Node
    # -------------------------------
    def api_execution_node(self, state):
        task = state.api_task
        caller = state.caller

        system_prompt = """
        You are a MOCK API SERVER, not a conversational assistant.
        Your job is to receive a JSON API call request and return a mock JSON response.
        Rules:
        - Always output VALID JSON only (no explanations, no natural language, no Markdown).
        - The top-level response must directly represent the result of the API call.
        - If parameters are missing, infer or provide sensible defaults.
        - Use realistic placeholder values depending on the API type.
        - Never invent extra keys outside what would logically exist.
        
        DONOT ATTEMPT TO EXECUTE ANY TOOL. YOU ARE A MOCK SERVER ONLY.
        """

        msg = safe_invoke_llm(self.llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(task))
        ])

        print(f"[APIExecutionNode] Mock API Response:\n{msg.content}")

        tool_msg = ToolMessage(
            content=msg.content,
            tool_call_id=task.get("name", "unknown_api") + str(uuid.uuid4()),
            additional_kwargs={"toolname": task.get("name", "unknown_api")},
            response_metadata={"type": "tool", "name": task.get("name", "unknown_api")}
        )

        state.chat_history.append(tool_msg)
        state.api_task = None
        state.task = msg.content
        state.next = caller
        state.caller = "api_execution"

        return state

    # -------------------------------
    # 3. Router Node
    # -------------------------------
    def route(self, state):
        """Pass-through router based on current state.next."""
        return state.next