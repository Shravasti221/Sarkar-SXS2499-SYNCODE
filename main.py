import json
from expert import Expert
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import json, os
from orchestrator import Orchestrator 
from simulators import problem_creator_llm, user_llm, user_route
from api_execution import APIPipeline
from utils.helpers import EventState
from utils.pydantic_objects import create_timestamped_file
#-------------------
# Build Workflow Graph
# -----------------------------
experts_json = json.load(open("experts.json")) 
workflow = StateGraph(EventState)
workflow.add_node("problem_creator", problem_creator_llm)
workflow.add_node("user", user_llm)

orchestrator = Orchestrator(experts = experts_json)
api_pipeline = APIPipeline()
workflow.add_node("orchestrator", orchestrator.node_fn)
workflow.add_node("api_execution", api_pipeline.process)

expert_nodes = {}
for name, definition in experts_json.items():
    expert = Expert(name, definition)
    expert_nodes[name] = expert
    workflow.add_node(name, expert.node_fn)

# # Graph edges as per your spec
workflow.add_edge(START, "problem_creator")
workflow.add_edge("problem_creator", "user")

for name, node in expert_nodes.items():
    workflow.add_conditional_edges(
        name,
        node.route,  # use the expert's own route method
        {
            "api_execution": "api_execution",
            "orchestrator": "orchestrator"
        }
    )
# ------------------------------Dynamic routing edges for API execution ------------------------------
aex_next_nodes = {expert_name: expert_name for expert_name in experts_json.keys()}
aex_next_nodes.update({
    "orchestrator": "orchestrator"
})
workflow.add_conditional_edges(
    "api_execution",
    api_pipeline.route,
    aex_next_nodes
)
# ------------------------------Dynamic routing edges for junior assistant ------------------------------
orchestrator_next_nodes = {expert_name: expert_name for expert_name in experts_json.keys()}
orchestrator_next_nodes.update({
    "user": "user",
    "api_execution": "api_execution"
})
workflow.add_conditional_edges("orchestrator", orchestrator.route, orchestrator_next_nodes)
# ------------------------------ ----------- User Route Edges ------------ ------------------------------
user_next_nodes = {
    "END": END,
    "orchestrator": "orchestrator",
}
workflow.add_conditional_edges("user", user_route, user_next_nodes)
# ---------------------------------------------------------------------------------------------------------

chain = workflow.compile()
display(Image(chain.get_graph().draw_mermaid_png()))

chat_history = []
state = EventState(chat_history=chat_history, ts = create_timestamped_file(), problem_created="Generate a problem statement based on the available experts and situation.")
state = chain.invoke(state)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
