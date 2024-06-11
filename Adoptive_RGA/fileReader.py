import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings,OpenAI,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)


class FileReader():
    DBpath: str = "faiss_index"
    folderName: str = "C:\\Users\\lawrencechen\\Desktop\\法律諮詢小幫手\\law-and-regulation-helper\\demo\\PDFfolder"
    vector_store:Any

    def __init__(self) -> Any:
        pdf_doc = self.folderReader()
        raw_text = self.get_pdf_text(pdf_doc)
        # doc_chunk = self.get_dirtectory_loader()
        text_chunks = self.get_text_chunks(raw_text)
        self.vector_store = self.get_vector_store(docs_chunks=text_chunks)
            
    def get_store(self):
        return self.vector_store

    def get_pdf_text(self,pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text= text+ page.extract_text()+ "\n"
        return  text

    def get_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=1250)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self,docs_chunks):
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(docs_chunks, embedding=embeddings)
        vector_store.save_local(self.DBpath)
        return vector_store.as_retriever()
    
    def get_dirtectory_loader(self):
        loader = PyPDFDirectoryLoader(self.DBpath, loader_cls=TextLoader)
        docs = loader.load_and_split()
        return docs
        

    def folderReader(self):
        paths = []
        path = f"{self.folderName}"
        for filename in os.listdir(path):
            if filename.endswith('.pdf'):  # Check if the file is a PDF
                pdf_path = os.path.join(path, filename)
                paths.append(pdf_path)
                print("no")
            else:
                print("yes")
                f = self.folderName+"\\"+filename
                if Path(f).is_dir():
                    print(f"----enter directory {f}----------")
                    for file in os.listdir(f):
                        if file.endswith('.pdf'):  # Check if the file is a PDF
                            pdf_path = os.path.join(f, file)
                            print(f"     -----{file}")
                            paths.append(pdf_path)

        # print(paths)
        if paths==[]:
            return False
        else:
            return paths

                
