import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings,OpenAI,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate



class FileReader():
    DBpath: str = "faiss_index"
    folderName: str = "C:\\Users\\lawrencechen\\Desktop\\法律諮詢小幫手\\law-and-regulation-helper\\demo\\PDFfolder"
    
    def __init__(self) -> None:
        pdf_doc = self.folderReader()
        if pdf_doc != False: 
            raw_text = self.get_pdf_text(pdf_doc)
            text_chunks = self.get_text_chunks(raw_text)
            self.get_vector_store(text_chunks)


    def get_pdf_text(self,pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text= text+ page.extract_text()+ "\n"
        return  text

    def get_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self,text_chunks):
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(self.DBpath)
    
    def folderReader(self):
        paths = []
        path = f"{self.folderName}"
        for filename in os.listdir(path):
            if filename.endswith('.pdf'):  # Check if the file is a PDF
                pdf_path = os.path.join(path, filename)
                paths.append(pdf_path)
        # print(paths)
        if paths==[]:
            return False
        else:
            return paths

                
