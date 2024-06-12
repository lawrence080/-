from langgraph.graph import END, StateGraph
from Graph_flow import GraphFlow
from Graph_state import GraphState

# from Index import indexing




class BuildGraph():
    workflow = StateGraph(GraphState)

    # def getFileInstance(instance):
    #     if instance!=None:
    #         return instance

    # fileReaderInstance = getFileInstance()
    GF: None
    def __init__(self, ret) -> None:
        self.GF = GraphFlow(ret)


    # Define the nodes
    def build(self, input):
        
        self.workflow.add_node("web_search", self.GF.web_search)  # web search
        self.workflow.add_node("retrieve", self.GF.retrieve)  # retrieve
        self.workflow.add_node("grade_documents", self.GF.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.GF.generate)  # generatae
        self.workflow.add_node("transform_query", self.GF.transform_query)  # transform_query

        # Build graph



        self.workflow.set_conditional_entry_point(
            self.GF.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.GF.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_conditional_edges(
            "generate",
            self.GF.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )



        # Compile
        app = self.workflow.compile().stream(input)
        return app