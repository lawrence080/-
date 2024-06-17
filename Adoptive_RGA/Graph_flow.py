from langchain.schema import Document
# from Index import indexing
import LLM
import Web_search_tool
from LLM import (
    RouteQuery,
    GradeDocuments,
    GradeHallucinations,
    GradeAnswer,
                 )
from pprint import pprint
from langchain_community.tools.tavily_search import TavilySearchResults
# from fileReader import FileReader
from PdfMinerFileReader import FileReader

web_search_tool = TavilySearchResults(k=3)


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
        else:
            vector_store = fileReader.getRegStore()
        documents = vector_store.invoke(question)
        return {"documents": documents, "question": question}
    

    def SPECvectorstore(self,state):
        return self.retrieve(state,"spec")

    def REGvectorstore(self,state):
        return self.retrieve(state,"reg")
    




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

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = LLM.question_reWriter().invoke({"question": question})
        return {"documents": documents, "question": better_question}


    def web_search(self,state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = Web_search_tool.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        print(" -----web done---- ")

        return {"documents": web_results, "question": question}


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
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
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
            score = GradeAnswer.answer_grade().invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"