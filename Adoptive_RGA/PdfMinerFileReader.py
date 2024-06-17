from langchain_community.document_loaders import PDFMinerLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings,OpenAI,ChatOpenAI
from langchain_community.vectorstores import FAISS





class FileReader():
    SpecfolderName:str = "C:\\Users\\lawrencechen\\Desktop\\法律諮詢小幫手\\law-and-regulation-helper\\demo\\PDFfolder\\specData"
    RegfolderName:str = "C:\\Users\\lawrencechen\\Desktop\\法律諮詢小幫手\\law-and-regulation-helper\\demo\\PDFfolder\\regData"
    DBpath: str = "faiss_index"
    SpecVectorStore:None = None
    RegVectorStore:None = None
    initialized:bool = False

    def __init__(self) -> None:
        if self.initialized:
            pass
        else:
            self.setDB()

    def setDB(self):
        specPath = self.specfolderReader()
        regPath = self.regfolderReader()
        specRetriever = self.loadPDFDoc(specPath,"SPEC")
        regRetriever = self.loadPDFDoc(regPath,"REG")
        self.SpecVectorStore = specRetriever
        self.RegVectorStore = regRetriever

    def getSpecStore(self):
        emb = OpenAIEmbeddings()
        return FAISS.load_local(self.DBpath,embeddings=emb,index_name="SPEC",allow_dangerous_deserialization=True).as_retriever()

    def getRegStore(self):
        emb = OpenAIEmbeddings()
        return FAISS.load_local(self.DBpath,embeddings=emb,index_name="REG",allow_dangerous_deserialization=True).as_retriever()

    def loadPDFDoc(self, docs, index):
        for pdfFilePath in docs:
            
            loader = PDFMinerLoader(pdfFilePath)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1250)
            chunk = loader.load_and_split(text_splitter=text_splitter)
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunk,embedding=embeddings)
            vector_store.save_local(self.DBpath, index_name=index)
        return vector_store.as_retriever()

    def specfolderReader(self):
        return self.folderReader(self.SpecfolderName)

    def regfolderReader(self):
        return self.folderReader(self.RegfolderName)

    def folderReader(self,name):
        paths = []
        path = f"{name}"
        for filename in os.listdir(path):
            if filename.endswith('.pdf'):  # Check if the file is a PDF
                pdf_path = os.path.join(path, filename)
                paths.append(pdf_path)
                print("no")
            else:
                print("yes")
                f = name+"\\"+filename
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
