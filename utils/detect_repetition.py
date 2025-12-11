from langchain_core.messages import BaseMessage

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0.0

def check_repetition(state, text: str, name: str) -> bool:
    """
    Analyze chat history to check if the current text is too similar to the last 5 responses from the same name.
    Returns False if Jaccard similarity > 80% with any of the last 5, else True.
    """
    same_name_msgs = [
        msg for msg in reversed(state.chat_history)
        if isinstance(msg, BaseMessage) and hasattr(msg, 'response_metadata') 
        and msg.response_metadata.get('name') == name
    ]
    

    last_5 = same_name_msgs[:5]
    
    for msg in last_5:
        sim = jaccard_similarity(text, msg.content)
        if sim > 0.8:
            return False
    
    return True