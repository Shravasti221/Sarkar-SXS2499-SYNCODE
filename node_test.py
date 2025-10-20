from langgraph.graph import StateGraph, START, END

def test_single_node(node_name: str, node_fn, state_class, test_state_obj):
    """
    Build and execute a minimal single-node LangGraph workflow
    for quick debugging of any individual node.

    Args:
        node_name (str): Name of the node (e.g., "clarification").
        node_fn (function): Node function to test (e.g., clarification_node).
        state_class (class): The state class used in your workflow (e.g., EventState).
        test_state_obj (object): An initialized instance of that state class.

    Returns:
        final_state (object): The resulting state after running the node.
    """

    # ------------------- Build Single-Node Workflow -------------------
    print(f"\n[TEST NODE: {node_name}] Building single-node workflow...")
    workflow = StateGraph(state_class)
    workflow.add_node(node_name, node_fn)
    workflow.add_edge(START, node_name)
    workflow.add_edge(node_name, END)
    chain = workflow.compile()

    # ------------------- Run Test -------------------
    print(f"[TEST NODE: {node_name}] Invoking node with test state...\n")
    final_state = chain.invoke(test_state_obj)

    # ------------------- Diagnostics -------------------
    print("\n[TEST NODE RESULT SUMMARY]")
    print(f"Next: {final_state.next}")
    print(f"Task: {getattr(final_state, 'task', None)}")

    if hasattr(final_state, "chat_history") and final_state.chat_history:
        print(f"Last message: {final_state.chat_history[-1].content}")
    else:
        print("No chat history in state.")

    print("\n--- Node Test Completed ---\n")
    return final_state
