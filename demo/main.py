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
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.getenv("LANGCHAIN_API_KEY")









def user_input(user_question, memory:dict):
    embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = chatBot.get_conversational_chain()

    
    response = chain.run(
        input_documents=docs, question=user_question
        , return_only_outputs=True)
    memory.update({"last_question": user_question, "last_answer": response})
    print(response)
    st.write("Reply: ", response)




def main():
    memory ={}
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI💁")
    fileReader.FileReader()
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, memory)





if __name__ == "__main__":
    main()