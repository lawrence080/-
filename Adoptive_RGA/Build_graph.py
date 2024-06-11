from langgraph.graph import END, StateGraph
import Graph_flow
from Graph_state import GraphState

# from Index import indexing



workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", Graph_flow.web_search)  # web search
workflow.add_node("retrieve", Graph_flow.retrieve)  # retrieve
workflow.add_node("grade_documents", Graph_flow.grade_documents)  # grade documents
workflow.add_node("generate", Graph_flow.generate)  # generatae
workflow.add_node("transform_query", Graph_flow.transform_query)  # transform_query

# Build graph



workflow.set_conditional_entry_point(
    Graph_flow.route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    Graph_flow.decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    Graph_flow.grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()