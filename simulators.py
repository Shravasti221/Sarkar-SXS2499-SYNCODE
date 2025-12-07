import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from utils.helpers import print_message
from utils.pydantic_objects import EventState, write_pydantic_object
# -----------------------------

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
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
    system_prompt = ("""
You are the Problem Creator for an Event Management Assistant.  
Your task is to generate exactly **one realistic and manageable problem scenario** based on the below trajectory.  """ 
    + state.problem_created["problem_statement"][0] + """"

Constraints:
1. Avoid catastrophic problems (e.g., full venue destruction, speaker cancellations affecting the whole event, massive technical failures). Focus on **common, everyday hiccups**.
2. Keep the problem description **concise (under 300 words)**.
3. Include **enough context** (event type, size, all required api parameter values and time) so assistants and tools can reason about a solution.
4. Give **exactly one problem scenario**.

Output format:
- Plain text **problem description only**.
""")
    msgs = [SystemMessage(content = system_prompt)]+ state.chat_history
    # print(chat_history_to_string(msgs))
    msg = llm.invoke(msgs)
    state.problem_statement = msg.content + "\n" + "\n".join([hint[2][0] + ": " + hint[2][1]  for hint in state.problem_created["hints"]])
    print_message(state, "ProblemCreatorLLM", msg.content)
    return state


# -----------------------------
# User node
# -----------------------------
def user_llm(state: EventState):
    """User simulates actions based on problem + chat history"""
    problem = state.problem_statement
    history_text = "Last 3 messages in conversation: \n"
    latest_message = ""
    if len(state.chat_history) > 0:
        pick_messages = min(3, len(state.chat_history))
        history_text = "\n".join([
            f"{m.response_metadata['type'].upper()}: {m.content}" for m in state.chat_history[-pick_messages:]
        ])
        latest_message = f"The latest message to be responded to is:\n{state.chat_history[-1].content}"
    else:
        history_text = "Explain the problem in first person to the AI experts."
    system_prompt = f"""
You are the USER in a simulated conversation with AI experts.

Context:
{problem}

Your behavior rules:
1. You are a normal person trying to solve the above problem.
2. Do NOT attempt to solve the problem yourself.
3. Read the chat history carefully and decide:
   - If your issue already seems resolved respond with "END OF CONVERSATION".
   - If the conversation indicates that your next steps require external actions, respond with "END OF CONVERSATION".
   - Otherwise, respond as a human who needs help, by expressing confusion or frustration naturally.
   - If you have been asked for clarification or more info, provide it based on the context.
4. Always speak in the first person (e.g., “I’m trying to…”).
5. Keep the tone realistic and concise — like an actual user chatting, not an AI.
6. Donot repeat the exact intent in the conversation. Instead, use your own words to convey the same information. 
7. DO NOT offer solutions
8. Generate relevant answers for the clarifications. FOr example if api key is asked, generate a relevant api key based on context. Similarly if relevant date, location, etc are requested for clarification, provide the same after looking at the `Context`.

{history_text}

{latest_message}
Output: a single user message in plain English.
    """
    msgs = [SystemMessage(content = system_prompt)] + state.chat_history
    msg = llm.invoke(msgs)
    
    print("SYSTEM PROMPT FOR USER LLM:\n", system_prompt)
    print("_______________________________________________________________")
    print_message(state, "User", msg.content)
    
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
    return state.next
    
