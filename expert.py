import json
import uuid
from utils.llm import llm
from utils.pydantic_objects import EventState, responseFormat, write_pydantic_object
from utils.safe_invoke_llm import safe_invoke_llm
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from utils.json_format import JsonFormat
from utils.detect_repetition import check_repetition
from utils.helpers import print_message
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
        sys_prompt, user_prompt = self.build_system_prompt(state)
        chat_history = [SystemMessage(content = sys_prompt), HumanMessage(content=user_prompt)] + state.chat_history
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
        print_message(state, f"Expert-{self.name}", formatted_output)
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
                print("[EXPERT] Expert requested for task to be executed but did not fill what api to call.\n")
                state.chat_history.append(ToolMessage(content=f"Error: Expert {self.name} requested API execution but did not provide task details. Raw Text: {msg.content}", tool_call_id= "missing_task_error" + str(uuid.uuid4()), response_metadata={"type": "error", "name": "missing_task_error"}))
                state.next = "orchestrator"
                return state
            state.api_task = expert_obj.task.model_dump()
            state.chat_history.append(AIMessage(content=f"Expert {self.name} requested API execution.", response_metadata={"type": "expert", "name": self.name}))
            state.caller = self.name
            state.next = "api_execution"
            return state
        if state.caller =="api_execution" and state.next != "api_execution":
            state.api_task = None
        if expert_obj.route == "response":
            response_text = expert_obj.response or ""
            print_message(state, f"[Response] Expert-{self.name}", response_text)
            state.chat_history.append(AIMessage(content=response_text, response_metadata={"type": "expert", "name": self.name}))
            state.caller = self.name
            state.next = "orchestrator"
            return state
        

    def build_system_prompt(self, state: EventState) -> str:
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
        sys_prompt_filler = ""
        if state.caller =="api_execution":
            if len(state.problem_created["hints"]):
                user_prompt_filler = f"The conversation is expected to flow in this manner: {state.problem_created["hints"][0]["hint"] if len(state.problem_created["hints"]) else "The conversation is expected to be complete after your interpretation of this api call."}"
            else:
                user_prompt_filler = "The conversation is expected to be complete after your interpretation of this api call."
            user_prompt = f"""The Previous API execution has been completed. Use the results obtained in your reasoning to respond as an expert in {self.name}. 
            Do not call the same function again and attempt to respond.
            {user_prompt_filler}
            Generate a response as an expert in {self.name}."""

            sys_prompt_filler = """- If the user’s request requires external data or action via an API, respond ONLY in the following JSON format:  
        - If you have sufficient data to respond, respond ONLY in the following JSON format:  
        ```
        {
          "route": "orchestrator",
          "task": null,
          "response": "<your English response here>"
        }
        ```"""
        else:
            user_prompt_filler = ""
            if len(state.problem_created["hints"]):
                user_prompt_filler = f"""The conversation is expected to flow in this manner: {state.problem_created["hints"][0]["hint"]}
                
                The expected API call looks like the below:
                {json.dumps(state.problem_created["hints"][0]["api_description"])}
                Before emitting the API call, follow these rules:
                1. Based on the conversation history, fill the parameters.
                2. Replace any placeholders (YOUR_API_KEY, YYYY-MM-DD, <value>, etc.) with actual values from the conversation context.
                3. If a parameter CANNOT be filled from context, respond to the orchestrator (asking user) requesting the correct details for all non-optional parameters.
                4. Do NOT invent any parameters or values.
                """
            
            user_prompt = f"""Based on the chat history, decide your next action or reply as the expert in {self.name}.
            {user_prompt_filler}
            Generate a response as an expert in {self.name}. """
            sys_prompt_filler = """- If the user’s request requires external data or action via an API, respond ONLY in the following JSON format:  
        ```
        {
          "route": "api_execution",
          "task": {
            "name": "<api_name>",
            "params": {
              "<param_name>": "<param_value>",
              "<param_name>": "<param_value>"
            }
          }
        }
        ```
        
        OR 
        
        - If you have sufficient data to respond, respond ONLY in the following JSON format:  
        ```
        {
          "route": "orchestrator",
          "task": null,
          "response": "<your English response here>"
        }
        ```"""
        # Build the final system prompt
        system_prompt = f"""
        {expert_prompt}
        
        Available API Tools:
        {apis_text}
        
        Instructions:  
        - Read the chat history carefully.  
        - If you can directly answer the user in English (based only on your expertise and available context), produce a short, clear English response.  
        {sys_prompt_filler}
        Rules:  
        1. Never mix English explanation with tool call JSON. Choose one.  
        2. Always match the available APIs and their parameter schema from context.  
        3. Be concise in English responses.  
        4. Do not invent APIs or parameters outside the provided context.  
        
        
        Final Output:  
        - Either plain English text in response json format
        - OR exactly one JSON object in the above format.
        - DONOT attempt to EXECUTE any tool.
        """
        return system_prompt.strip(), user_prompt.strip()


    def route(self, state: EventState):
        """Decide whether to go to API or back to orchestrator"""
        write_pydantic_object(state, state.ts)
        input()
        return state.next