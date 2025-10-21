from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
# -----------------------------
# LLM Initialization
# -----------------------------
if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY not found")

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    # reasoning_effort="low",
     model_kwargs={
        "stream": False,
        "tool_choice": "none",  # Explicitly disable tool calls
        # Optionally enforce JSON output
        # "response_format": {"type": "json_object"}
    })
