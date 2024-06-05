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

load_dotenv()
os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.getenv("LANGCHAIN_API_KEY")







def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    根據提供的上下文盡可能詳細地回答問題，確保提供所有細節。如果答案不在提供的上下文中，不要捏造答案，請回答不知道。\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # combinePrompt = """
    #     根據提供的上下文盡可能詳細地回答問題，確保提供所有細節。如果在上下文中有出現先相關的資訊請如實回答，不要說不知道。如果答案不在提供的上下文中，不要捏造答案，請回答不知道。
    #     summeries:\n {summaries}\n
    #     Answer:
    # """

    model = ChatOpenAI(model="gpt-3.5-turbo-0125",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    combinePrompt = map_reduce_prompt_v2.COMBINE_PROMPT_SELECTOR.get_prompt(model)
    chain = load_qa_chain(model, chain_type="map_reduce",question_prompt= prompt, combine_prompt=combinePrompt)

    return chain



def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question,k=8)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()