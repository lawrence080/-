from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import Index




# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    # LLM with function call

    def router():
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to 電信號碼申請，電信號碼使用收費標準，無線電頻率及特殊電訊號使用收費標準以及網際網路管理業務.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | structured_llm_router
        return question_router




class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    def retrieval_grader():
        # LLM with function call
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """你是一位評分員，負責評估檢索到的文件與使用者問題的相關性。\n
                如果文件包含與使用者問題相關的關鍵字或語義，則將其評定為相關。\n
                這不需要嚴格的測試。目標是篩選出錯誤的檢索結果。\n
                給出一個「yes」或「no」的二元分數，以指示文件是否與問題相關。
                """
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader
        return retrieval_grader


from langchain import hub
from langchain_core.output_parsers import StrOutputParser
def generat():
# Prompt
    # prompt = hub.pull("rlm/rag-prompt")
    promptTemplate = """

        你是一個回答問題的助理。使用以下檢索到的上下文來回答問題。如果你不知道答案，就直接說你不知道。

        Question:{question}

        Context:{context}

        Answer: 

    """
    prompt = PromptTemplate(template=promptTemplate, input_variables=["question","context"])
    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

    def hallucination_grader():

        # LLM with function call
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)

        # Prompt
        system = """你是一位評分員，負責評估大型語言模型（LLM）生成的答案是否基於／支持於一組檢索到的事實。\n 
    給予「yes」或「no」的二元分數。「yes」表示答案是基於／支持於這組事實。"""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        hallucination_grader = hallucination_prompt | structured_llm_grader
        return hallucination_grader


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

    def answer_grade():
        # LLM with function call
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """你是一位評分員，負責評估一個答案是否解決了問題。\n
                給予「yes」或「no」的二元分數。「yes」表示答案解決了問題。"""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | structured_llm_grader
        return answer_grader


def question_reWriter():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Prompt
    system = """你是一位問題重寫員，負責將輸入的問題轉換為針對向量庫檢索優化的更佳版本。\n
    查看輸入問題並嘗試推理其背後的語義意圖／含義。"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter