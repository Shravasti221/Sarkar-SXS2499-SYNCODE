import json
import uuid
from llm import llm
from pydantic_objects import EventState, responseFormat, writePydanticObject
from safe_invoke_llm import safe_invoke_llm
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from utils.json_format import JsonFormat
from utils.detect_repetition import check_repetition
# -----------------------------
# Expert Node
# -----------------------------
class Expert:
    def __init__(self, name: str, definition: dict):
        self.name = name
        self.description = definition.get("description", "")
        self.expert_prompt = definition.get(
            "expert_prompt",
            f"You are {self.name}, expert in {self.description}. Answer clearly."
        )
        
        self.json_format = JsonFormat(
            pydantic_model=responseFormat,
            expected_format="""
            {
              "route": "response" or "api_execution" or "<expert_name>",
              "task": None or { +""" + json.dumps(responseFormat.model_json_schema(), indent=2) + """
              } or null,
              "response": "<string or null>"
            }
            """,
            max_retries=3,
            verbose=True
        )
        
        self.tools = definition.get("apis", {})

    def node_fn(self, state: EventState):
        sys_prompt = self.build_system_prompt()
        chat_history = [SystemMessage(content = sys_prompt)]+ state.chat_history
        msg = safe_invoke_llm(llm, chat_history)
        attempts = 0
        max_retries = 3

        while attempts < max_retries:
            if check_repetition(state, msg.content, self.name):
                break
            
            attempts += 1
            print(f"[REPETITION DETECTED - Retry {attempts}/{max_retries}]")
            
            if attempts < max_retries:
                updated_sysprompt = sys_prompt+ f"\n\n[CRITICAL: Generate COMPLETELY NEW response. NO repetition of previous answers.]"
                msg = safe_invoke_llm(llm, [SystemMessage(content=updated_sysprompt)] + chat_history)
            else:
                print("[GIVING UP AFTER 3 RETRIES - Using last response]")
        
        formatted_output = self.json_format.refine(msg.content)
        try:
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
            state.api_task = expert_obj.task.model_dump()
            state.caller = self.name
            state.next = "api_execution"
            return state
        if expert_obj.route == "response":
            response_text = expert_obj.response or ""
            state.chat_history.append(AIMessage(content=response_text, response_metadata={"type": "expert", "name": self.name}))
            state.caller = self.name
            state.next = "orchestrator"
            return state
        


    def build_system_prompt(self):
        """
        Build a structured system prompt for an agent that can either respond in English
        or call an API with a JSON object.
        
        Args:
            role_name (str): Name of the agent role (e.g., "LogisticsManager").
            expert_prompt (str): Domain-specific instructions for the role.
            apis (dict): Dictionary of available APIs and their params.
            chat_history (list[dict]): List of chat history messages, 
                                       e.g. [{"role": "user", "content": "message"}]
        
        Returns:
            str: Formatted system prompt string.
        """
        expert_prompt = self.expert_prompt
        apis = self.tools
        # Format APIs in a readable way
        api_descriptions = []
        for api_name, api_info in apis.items():
            params = api_info.get("params", {})
            param_list = ", ".join(f"{k}: {v}" for k, v in params.items())
            api_descriptions.append(f"- {api_name}: requires params {{{param_list}}}")
        apis_text = "\n".join(api_descriptions) if api_descriptions else "None available."
    
        # Build the final system prompt
        system_prompt = f"""
        {expert_prompt}
        
        Available API Tools:
        {apis_text}
        
        Instructions:  
        - Read the chat history carefully.  
        - If you can directly answer the user in English (based only on your expertise and available context), produce a short, clear English response.  
        - If the user’s request requires external data or action via an API, respond ONLY in the following JSON format:  
        ```
        {{
          "route": "api_execution",
          "task": {{
            "name": "<api_name>",
            "params": {{
              "<param_name>": "<param_value>",
              "<param_name>": "<param_value>"
            }}
          }}
        }}
        ```
        Rules:  
        1. Never mix English explanation with tool call JSON. Choose one.  
        2. Always match the available APIs and their parameter schema from context.  
        3. Be concise in English responses.  
        4. Do not invent APIs or parameters outside the provided context.  
        5. When in doubt, default to a normal English response.  
        
        
        Final Output:  
        - Either plain English text (answer to user)  
        - OR exactly one JSON object in the above format.
        - DONOT attempt to EXECUTE any tool.
        """
        return system_prompt.strip()


    def route(self, state: EventState):
        """Decide whether to go to API or back to orchestrator"""
        writePydanticObject(state, state.ts)
        print("________________________________________________________________________________________________________")
        return state.next