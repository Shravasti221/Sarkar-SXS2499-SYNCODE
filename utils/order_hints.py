import re

def extract_and_order_steps(problem_data: dict, problem_index: int = 0):
    """
    Extract sentences for API calls, fill steps_with_hints,
    and return a list sorted so start_dialogue is ALWAYS first.
    """

    # ---- Extract & clean text ----
    text = problem_data["problem_statement"][problem_index]
    cleaned = re.sub(r"\s+", " ", text)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    # ---- Build list of all API names ----
    api_list = [step_name for step_name, _ in problem_data["steps"]]
    api_list.append("start_dialogue")

    # ---- Map API → sentences ----
    api_to_sentence = {}
    for api in api_list:
        pattern = re.compile(rf"\({api}\)", re.IGNORECASE)
        api_to_sentence[api] = [s.strip() for s in sentences if pattern.search(s)]

    # ---- Build unsorted results ----
    results = []
    for key in problem_data["steps_with_hints"]:
        index, prev_api, curr_api = key
        extracted = api_to_sentence.get(curr_api, [])
        hint_sentence = " ".join(extracted) if extracted else ""
        results.append((index, prev_api, curr_api, hint_sentence))

    # ---- Custom sorting: start_dialogue always first ----
    def sort_key(item):
        idx, prev_api, curr_api, _ = item
        if prev_api == "start_dialogue":
            return (-1, 0)  # forces it to be first
        return (idx, 1)

    results.sort(key=sort_key)
    return results