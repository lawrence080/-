from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel




# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["SPECvectorstore", "REGvectorstore"] = Field(
        ...,
        description="Given a user question choose to route it to special_vectorstore which contains info about 各頻段的用途及應用, or a regular_vectorstore which contains general infos.",
    )
    # LLM with function call


    def router():
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to vectorstores of different catagories.
        The SPECvectorstore contains documents related to 各頻段的用途及應用.
        Use the SPECvectorstore for questions on 各頻段的用途及應用. Otherwise, use REGvectorstore."""
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
        description="Documents are relevant to the question, 'yes' or 'no'. if there are part similar or related say yes"
    )
    def retrieval_grader():
        # LLM with function call
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """你是一位評分員，負責評估檢索到的文件與使用者問題的相關性。\n
                如果文件包含與使用者問題相關的關鍵字或語義，則將其評定為相關。\n
                這不需要嚴格的測試。目標是篩選出錯誤的檢索結果。\n
                給出一個「yes」或「no」的二元分數，以指示文件是否與問題相關。\n

                example:
                    question: 跟我說500的頻段有哪些?
                    documents:  530-536,
                                542-548,
                                554-560,
                                566-572,
                                578-584,
                                590-596*3
                                供無線電
                                視使用，
                                執照期間
                                為 9 年，
                                期滿應申
                                請辦理換
                                發。
                    answer: yes
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

        你是一個回答問題的助理。使用以下檢索到的上下文來回答問題。提供越多信息越好，請盡量仔細的給出所有資訊。如果你不知道答案，就直接說你不知道。請用繁體中文回答。你的回答必須小於10000個token。

        Question:{question}

        Context:{context}

        Answer: 

    """
    prompt = PromptTemplate(template=promptTemplate, input_variables=["question","context"])
    # LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
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
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
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


class agentAnalizer():
    pass

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

    def answer_grade():
        # LLM with function call
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
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
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.4)

    # Prompt
    system = """你是一位問題重寫員，負責將輸入的問題轉換為針對向量庫檢索優化的更佳版本。\n
    查看輸入問題並嘗試推理其背後的語義意圖／含義。請用繁體中文回答"""
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