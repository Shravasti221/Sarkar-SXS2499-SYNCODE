import json
import time
def safe_invoke_llm(llm, messages, max_retries=3, retry_delay=3, **kwargs):
    """
    Invokes the LLM with built-in retry on Groq rate limit responses.
    Handles both structured JSON errors and Python exceptions.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages, **kwargs)
            # Check if response looks like a JSON error blob
            try:
                data = json.loads(response.content)
                if isinstance(data, dict) and "error" in data:
                    err = data["error"]
                    if isinstance(err, dict) and err.get("code") == "rate_limit_exceeded":
                        print(f"[Attempt {attempt}] Rate limit hit — retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
            except Exception:
                # Not a JSON string, proceed as normal
                pass

            # Normal success path
            return response

        except Exception as e:
            # Some APIs raise directly instead of returning a JSON error
            msg = str(e)
            if "rate limit" in msg.lower() or "rate_limit_exceeded" in msg.lower():
                print(f"[Attempt {attempt}] Exception: Rate limit reached — retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                # Non-rate-limit exception — stop retrying
                raise

    raise RuntimeError(f"LLM failed after {max_retries} attempts due to repeated rate limit errors.")

