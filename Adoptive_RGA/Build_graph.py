from langgraph.graph import END, StateGraph

from Graph_state import GraphState
from langgraph.checkpoint.memory import MemorySaver
import asyncio
# from Index import indexing
from PdfMinerFileReader import FileReader
from pprint import pprint
import LLM
from LLM import (
    RouteQuery,
    GradeDocuments,
    GradeHallucinations,
    GradeAnswer,
                 )



class BuildGraph():
    recursionLimit: int = 2
    workflow = StateGraph(GraphState)
    memery = MemorySaver()
    # def getFileInstance(instance):
    #     if instance!=None:
    #         return instance

    GF: None
    vectorStore: None

    def __init__(self) -> None:
        # if not FileReader.initialized:
        #     self.vectorStore = FileReader()
        #     FileReader.initialized = True
        self.GF = GraphFlow()

    def __del__(self):
        print('Destructor called, Employee deleted.')


    # Define the nodes
    def build(self,input):
        self.workflow.add_node("REGvectorstore", self.GF.REGvectorstore)  # retrieve
        self.workflow.add_node("SPECvectorstore", self.GF.SPECvectorstore)
        self.workflow.add_node("BOTHvectorstore", self.GF.BOTHvectorstore)
        self.workflow.add_node("grade_documents", self.GF.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.GF.generate)  # generatae
        self.workflow.add_node("transform_query", self.GF.transform_query)  # transform_query
        self.workflow.add_node("recursion_limit_exceed",self.GF.recursion_limit_exceed)

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
                "recursion_limit_exceed":"recursion_limit_exceed",
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        # self.workflow.add_conditional_edges(
        #     "transform_query",
        #     self.GF.route_question,
        #     {
        #         "SPECvectorstore": "SPECvectorstore",
        #         "REGvectorstore": "REGvectorstore",
        #     }
        # )
        self.workflow.add_edge("transform_query","BOTHvectorstore")
        self.workflow.add_edge("BOTHvectorstore", "grade_documents")
        self.workflow.add_conditional_edges(
            "generate",
            self.GF.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "recursion_limit_exceed":"recursion_limit_exceed",
            },
        )
        self.workflow.set_finish_point("recursion_limit_exceed")
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
    



class GraphFlow():
    retriever:None

    def __init__(self) -> None:
        pass

    def retrieve(self,state,type):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        # fileReader = instance
        fileReader = FileReader()
        if type == "spec":
            vector_store = fileReader.getSpecStore()
            documents = vector_store.invoke(question)
        elif type == "reg":
            vector_store = fileReader.getRegStore()
            documents = vector_store.invoke(question)
        else:
            vector_store = fileReader.getRegStore()
            vector_store2 = fileReader.getSpecStore()
            documents = vector_store.invoke(question) + vector_store2.invoke(question)
        return {"documents": documents, "question": question}
    

    def SPECvectorstore(self,state):
        return self.retrieve(state,"spec")

    def REGvectorstore(self,state):
        return self.retrieve(state,"reg")
    

    def BOTHvectorstore(self,state):
        print("---USING BOTH---")
        return self.retrieve(state,"both")
    




    def generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # RAG generation
        generation = LLM.generat().invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def recursion_limit_exceed(self,state):
        BuildGraph.recursionLimit = 3
        question = state["question"]
        generation = "很抱歉，我在文件中找不到相關的資訊，請提供更詳細的資訊或換個方式提問，謝謝"
        return {"documents": [], "question":question , "generation": generation}

    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = GradeDocuments.retrieval_grader().invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # filtered_docs.append(d)
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        BuildGraph.recursionLimit -= 1
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = LLM.question_reWriter().invoke({"question": question})
        return {"documents": documents, "question": better_question}


    # def web_search(self,state):
    #     """
    #     Web search based on the re-phrased question.

    #     Args:
    #         state (dict): The current graph state

    #     Returns:
    #         state (dict): Updates documents key with appended web results
    #     """

    #     print("---WEB SEARCH---")
    #     question = state["question"]

    #     # Web search
    #     docs = Web_search_tool.web_search_tool.invoke({"query": question})
    #     web_results = "\n".join([d["content"] for d in docs])
    #     web_results = Document(page_content=web_results)
    #     print(" -----web done---- ")

    #     return {"documents": web_results, "question": question}


    ### Edges ###


    def route_question(self,state):
        """
        Route question to general or special.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = RouteQuery.router().invoke({"question": question})
        print(source)
        if source.datasource == "SPECvectorstore":
            print("---ROUTE QUESTION TO SPECvectorstore---")
            return "SPECvectorstore"
        elif source.datasource == "REGvectorstore":
            print("---ROUTE QUESTION TO REGvectorstore---")
            return "REGvectorstore"


    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            if BuildGraph.recursionLimit>0:
                print(
                    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
                )
                return "transform_query"
            else:
                print("---RECURSION LIMIT HAS EXCEED---")
                return "recursion_limit_exceed"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
    
    

    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = GradeHallucinations.hallucination_grader().invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            # score = GradeAnswer.answer_grade().invoke({"question": question, "generation": generation})
            # grade = score.binary_score
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if BuildGraph.recursionLimit>0:
                pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"
            else:
                print("---RECURSION LIMIT HAS EXCEED---")
                return "recursion_limit_exceed"