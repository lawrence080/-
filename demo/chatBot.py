import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings,OpenAI,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import map_reduce_prompt_v2
import fileReader
import chatBot



load_dotenv()
os.getenv("OPENAI_API_KEY")

def get_conversational_chain():

    prompt_template = """
    根據提供的上下文盡可能詳細地回答問題，確保提供所有細節。如果答案不在提供的上下文中，請不要回答!\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    # combinePrompt = """
    #     根據提供的上下文盡可能詳細地回答問題，確保提供所有細節。如果在上下文中有出現先相關的資訊請如實回答，不要說不知道。如果答案不在提供的上下文中，不要捏造答案，請回答不知道。
    #     summeries:\n {summaries}\n
    #     Answer:
    # """

    model = ChatOpenAI(model="gpt-3.5-turbo-0125",
                             temperature=0.1)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    combinePrompt = map_reduce_prompt_v2.COMBINE_PROMPT_SELECTOR.get_prompt(model)
    chain = load_qa_chain(model, chain_type="map_reduce",question_prompt= prompt, combine_prompt=combinePrompt)

    return chain