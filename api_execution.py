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
        for k, v in data.items():
            all_apis = {}
            for api in v["apis"]:
                all_apis[api["APIName"]] = api
            data[k]["apis"] = all_apis

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
    def clarification_node(self, state: EventState)-> EventState:
        task = state.api_task
        caller = state.caller

        # print(f"\n[ClarificationNode] Incoming task from: {caller}")
        # print(f"[ClarificationNode] Raw task:\n{json.dumps(task, indent=2)}")
        print(f"[ClarificationNode] From: {caller}:{task}")

        system_prompt = f"""
        You are a strict API Parameter Validator & Clarifier.

        Rules:
        1. Verify that every params required by the API are present and there are no null/placeholders for non-optional params. Optional params can be missing or null.
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
        2. If ANY param is a placeholder (YOUR_API_KEY, YYYY-MM-DD, <value>, etc.) → status="fail". Mention all such placeholder parameters in response.
        3. If a parameter is already clarified, DONOT ASK FOR IT AGAIN.
        4. ONLY JSON and no other text.
        """
        
        user_prompt = f"""The expected API call structure is:
        {json.dumps(state.api_task, indent=2)}
        Validate the parameters strictly according to the rules."""

        conversation_text = "\n".join([
            f"{m.type.upper()}: {m.content}" for m in state.chat_history[-10:]
        ])

        input_payload = {"api_task": task, "context": conversation_text}

        msg = safe_invoke_llm(self.llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content= user_prompt + str(json.dumps(input_payload)))
        ])

        print(f"[ClarificationNode] LLM Output:\n{msg.content}")

        try:
            match = re.search(r'{.*}', msg.content, re.DOTALL)
            if not match:
                result = {"status": "fail", "issue": "No JSON object found in clarifier response." + str(msg.content)}
                # raise ValueError("No JSON object found.")
            if match:
                clarification = match.group(0)
            else:
                clarification = "\"status\": \"fail\", \"issue\": \"No JSON object found in clarifier response. Current Feedback:" +msg.content +"}"
            result = json.loads(clarification)
        except Exception as e:
            print(f"[ClarificationNode] JSON Parse Error: {e}")
            result = {"status": "fail", "issue": "Non-JSON response from clarifier."}

        if result.get("status") == "ok":
            print("[ClarificationNode] Parameters validated")
            
        elif result.get("status") == "fail":
            print("[ClarificationNode] Parameters need clarification:", result.get("response"))
            feedback = f"Parameter issue detected: {result.get('response')}"
            updated_history = state.chat_history + [AIMessage(content=feedback, response_metadata={"type": "expert", "name": "api_clarification"})]
            state.chat_history = updated_history
            state.next = "orchestrator"
            state.caller = caller
        else:
            issue = result.get("response", "Could not extract JSON string from msg content: ."+ msg.content)
            print("[ClarificationNode] Irrecoverable issue:", issue)
            feedback = f"Cannot execute {task.get('name', 'unknown')}: {issue}"
            updated_history = state.chat_history + [AIMessage(content=feedback, response_metadata={"type": "expert", "name": "api_clarification"})]
            state.chat_history = updated_history
            state.next = "orchestrator"
            state.caller = caller

        return state

    # -------------------------------
    # 2. API Execution Node
    # -------------------------------
    def api_execution_node(self, state):
        task = state.api_task
        caller = state.caller

        system_prompt = """
        You are a MOCK API SERVER, not a chatbot.
        Your job is to receive a JSON API call request and return a mock JSON response.
        Rules:
        - Always output VALID JSON only (no explanations, no natural language, no Markdown).
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
            tool_name=task.get("name", "unknown_api"),
            response_metadata={"type": "tool", "name": task.get("name", "unknown_api")}
        )

        state.chat_history.append(tool_msg)
        try:
            parsed_api_response = json.loads(msg.content)
        except json.JSONDecodeError:
            parsed_api_response = {"error": "Mock API returned invalid JSON format", "raw": msg.content}
            
        if (state.api_task.get("name") == state.problem_created["hints"][0]["api_name"]):
                # remove 0th index of hints[0]["api_name"]
                state.problem_created["hints"].pop(0)
                state.problem_created["steps"].pop(0)
                print(f"Hey there! I think we just ran a function. So I'll be tossing out {state.api_task.get("name")} from the hints now.\n Find the pending steps below!: ")
                for i in range(len(state.problem_created["hints"])):
                    print(state.problem_created["hints"][i]["api_name"])
  
            
        state.api_task = parsed_api_response
        state.next = caller
        state.caller = "api_execution"
        print(f"\n\n [STATE UPDATES AT API EXEC NODE]: state.next: {state.next}, state.caller: {state.caller} ")
        return state

    def route(self, state):
        """Pass-through router based on current state.next."""
        input()
        return state.next