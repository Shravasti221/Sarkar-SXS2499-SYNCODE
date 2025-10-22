import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from utils.helpers import print_message
from utils.pydantic_objects import EventState, writePydanticObject
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
    system_prompt = system_prompt = """
You are the Problem Creator for an Event Management Assistant.  
Your task is to generate exactly **one realistic and manageable problem scenario** that an event management team might face.  

Constraints:
1. The event must be **current or upcoming within the next 7 days**, e.g., conferences, concerts, weddings, corporate meetings, sports events, or community gatherings.
2. The problem should be **small to medium in scale** — it should be solvable using digital assistance tools like scheduling, logistics coordination, vendor communication, ticketing, or guest management.
3. Avoid catastrophic problems (e.g., full venue destruction, speaker cancellations affecting the whole event, massive technical failures). Focus on **common, everyday hiccups** such as:
   - Minor scheduling conflicts
   - Late vendor responses
   - Incorrect attendee information
   - Confusing sign-up forms
   - Small technical glitches (projector, microphone, Wi-Fi)
   - Minor miscommunications between staff or volunteers
4. Keep the problem description **concise (under 150 words)**.
5. Include **enough context** (event type, size, and time) so assistants and tools can reason about a solution.
6. Give **exactly one problem scenario**.

Output format:
- Plain text **problem description only**.  
- **Do NOT include solutions, next steps, hints, or commentary**.
"""
    msgs = [SystemMessage(content = system_prompt)]+state.chat_history
    # print(chat_history_to_string(msgs))
    msg = llm.invoke(msgs)
    state.problem_created = msg.content
    print_message(state, "ProblemCreatorLLM", msg.content)
    return state


# -----------------------------
# User node
# -----------------------------
def user_llm(state: EventState):
    """User simulates actions based on problem + chat history"""
    problem = state.problem_created
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
   
   - Otherwise, respond as a human who needs help, by:
     • Asking clarifying questions,
     • Expressing confusion or frustration naturally.
4. Always speak in the first person (e.g., “I’m trying to…”).
5. Keep the tone realistic and concise — like an actual user chatting, not an AI.
6. Generate one line at a time to simulate the user’s message
7. Donot hallucinate information that is not provided in the intent.
8. Donot repeat the exact intent in the conversation. Instead, use your own words to convey the same information. 
9. Donot repeated use the exact same phrases in the conversation. Instead, use synonyms or rephrase your sentences or ask something different but relevant.
10. DO NOT offer help or solutions.

Output: a single user message in plain English.
    """
    msgs = [SystemMessage(content = system_prompt)] + state.chat_history
    msg = llm.invoke(msgs)
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
    writePydanticObject(state, state.ts)
    return state.next
    
