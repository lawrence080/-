from langchain_community.document_loaders import PDFMinerLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings,OpenAI,ChatOpenAI
from langchain_community.vectorstores import FAISS





class FileReader():
    folderName:str = "C:\\Users\\lawrencechen\\Desktop\\法律諮詢小幫手\\law-and-regulation-helper\\demo\\PDFfolder"
    DBpath: str = "faiss_index"
    vectorStore:None

    def __init__(self) -> None:
        path = self.folderReader()
        retriver = self.loadPDFDoc(path)
        self.vectorStore = retriver


    def get_store(self):
        return self.vectorStore

    def loadPDFDoc(self, docs):
        for pdfFilePath in docs:
            
            loader = PDFMinerLoader(pdfFilePath)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=1250)
            chunk = loader.load_and_split(text_splitter=text_splitter)
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunk,embedding=embeddings)
            vector_store.save_local(self.DBpath)
        return vector_store.as_retriever()

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
