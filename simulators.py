import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from utils.helpers import print_message
from utils.pydantic_objects import EventState, write_pydantic_object
# -----------------------------


llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    stream= False,
    )
# -----------------------------
# Problem Creator node
# -----------------------------
def problem_creator_llm(state: EventState):
    # Reads the message in state.problem_created (and chat history if any) and generates a problem scenario
    # Updates the generated problem to state.problem_created
    
    print("\n[ProblemCreatorLLM] Generating problem scenario... :", state.problem_created["problem_statement"][0])
    system_prompt = """You are a story editor who helps in making corrections to scripts. 

Always stick to the expected function call order. The order of functions is the skeleton of the scene. If anything seems logically incorrect, make small changes to the story to make it viable.

Always output the problem in future tense, indicating actions to be done.
Do not extend or hallucinate any new actions or story points.

Output format: Plain Text
"""

    user_prompt = f"""
This is the suggested situation with the api calls: {"-> ".join([i[0] for i in state.problem_created["steps"]])} \n\n {state.problem_created["problem_statement"][0]}
Rephrase the above into a concise starting problem for a user facing the issue.
\n\nConstraints:
1. Avoid catastrophic problems (e.g., full venue destruction, speaker cancellations affecting the whole event, massive technical failures). Focus on **common, everyday hiccups**.
2. Keep the problem description **concise (under 300 words)**.
3. Include **enough context** (event type, size, etc) so assistants and tools can reason about a solution.
4. Give **exactly one problem scenario**.

Output format:
- Plain text **problem description only**.
"""
    msgs = [SystemMessage(content = system_prompt), HumanMessage(content = user_prompt)]
    # print(chat_history_to_string(msgs))
    msg = llm.invoke(msgs)
    state.problem_statement = msg.content
    print("----------------------------------------------------------------------------------------------")
    print_message(state, "ProblemCreatorLLM", msg.content)
    return state


# -----------------------------
# User node
# -----------------------------
def user_llm(state: EventState):
    """User simulates actions based on problem + chat history"""
    problem = state.problem_statement    
    if len(state.problem_created["hints"]) == 0:
        state.next= "END"
        
        print("!!____________________________________________________________________!!")
        print("Conversation ended by User.")
        print("!!____________________________________________________________________!!")
        return state
    
    system_prompt = f"""
    
You are the USER in a simulated conversation with AI experts.

Your behavior rules:
1. You are the person responsible for solving the Scene Description below.
2. Do NOT attempt to solve the problem yourself, instead talk to the AI assistants to get help.
3. Read the chat history carefully and decide:
   - If scenario is complete respond with "END OF CONVERSATION".
   - If you have been asked a question and you have the answer in the scene description or hint, provide it. Clarify intent.
4. Always speak in the first person (e.g., “I’m trying to…”).
5. Keep the tone realistic and concise — like an actual user chatting, not an AI.
6. Do not repeat the exact intent in the conversation. Rephrase.
7. Generate relevant answers for the clarifications. For example if api key is asked, generate a completely random API key or pick one from the problem statement. Similarly if relevant date, location, etc are requested for clarification, provide the same after looking at the `Context`.
8. DONOT mention function names
9. DONOT give clear instructions

Output: a single user message in plain English.

Scene Description:
{problem}
    """
    
    
    user_prompt = f"""
Next steps you can expect from the assistant:
{state.problem_created["hints"][0]["hint"]}
If asked for API key reply with the following key: {state.problem_created["hints"][0]["api_key"]}

Chat History
{"\n".join([f"{m.type.upper()}: {m.content}" for m in state.chat_history])}
You, the user, are the only one who can see the problem description and the context. So always explain the situation in case 
Recreate how the user would respond in this scenario.
    """
    msgs = [SystemMessage(content = system_prompt)] + [HumanMessage(content = user_prompt)]
    msg = llm.invoke(msgs)
    print_message(state, "user", msg.content)
    
    if "end of conversation" in msg.content.lower():
        state.next= "END"
        print("!!____________________________________________________________________!!")
        print("Conversation ended by User.")
        print("!!____________________________________________________________________!!")
    else:
        state.next= "orchestrator"
    state.chat_history.append(HumanMessage(content=msg.content, response_metadata={"type": "user", "name": "User"}))
    return state

def user_route(state: EventState) -> str:
    """Decides next node after User"""
    write_pydantic_object(state, state.ts)
    input()
    return state.next
    
