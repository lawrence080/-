from langgraph.graph import END, StateGraph
from Graph_flow import GraphFlow
from Graph_state import GraphState
from langgraph.checkpoint.memory import MemorySaver
import asyncio
# from Index import indexing
from PdfMinerFileReader import FileReader



class BuildGraph():
    workflow = StateGraph(GraphState)
    memery = MemorySaver()
    # def getFileInstance(instance):
    #     if instance!=None:
    #         return instance

    GF: None
    vectorStore: None

    def __init__(self) -> None:
        if not FileReader.initialized:
            self.vectorStore = FileReader()
            FileReader.initialized = True
        self.GF = GraphFlow()

    def __del__(self):
        print('Destructor called, Employee deleted.')


    # Define the nodes
    def build(self,input):
        self.workflow.add_node("REGvectorstore", self.GF.REGvectorstore)  # retrieve
        self.workflow.add_node("SPECvectorstore", self.GF.SPECvectorstore)
        self.workflow.add_node("grade_documents", self.GF.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.GF.generate)  # generatae
        self.workflow.add_node("transform_query", self.GF.transform_query)  # transform_query

        # Build graph



        self.workflow.set_conditional_entry_point(
            self.GF.route_question,
            {
                "SPECvectorstore": "SPECvectorstore",
                "REGvectorstore": "REGvectorstore",
            },
        )
        # self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("REGvectorstore", "grade_documents")
        self.workflow.add_edge("SPECvectorstore", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.GF.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_conditional_edges(
            "transform_query",
            self.GF.route_question,
            {
                "SPECvectorstore": "SPECvectorstore",
                "REGvectorstore": "REGvectorstore",
            }
        )
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
        # app = self.workflow.compile(checkpointer=self.memery).stream(input)
        app = self.workflow.compile(checkpointer=self.memery)
        coro = app.ainvoke(input, {"configurable": {"thread_id": "thread-1"}})
        return asyncio.run(coro)


    def compile(self, input):
        # Compile
        app = self.workflow.compile(checkpointer=self.memery)
        coro = app.ainvoke(input, {"configurable": {"thread_id": "thread-1"}})
        return asyncio.run(coro)