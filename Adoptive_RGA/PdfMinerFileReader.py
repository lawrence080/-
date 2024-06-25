from langchain_community.document_loaders import PDFMinerLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings,OpenAI,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal



class FileReader():
    """
        this class handle all the pdf file loading functionalities 

        NewFolderName: the path to newFolder, which is a folder for new user input files, 
                    after the files are loaded into the local vector database, all the pdf files in Newfolder will be moved to 
                    existedFile Folder 
        DBpath: the path to local vector database
    
    """
    NewFolderName:str = "Adoptive_RGA/PDFfolder/newData"
    DBpath: str = "Adoptive_RGA/faiss_index"
    initialized:bool = True

    def __init__(self) -> None:
        if self.initialized:
            pass
        else:
            self.addFileToVectorStore()

   
        

    def addFileToVectorStore(self):
        """
            helper function of loadPDFDoc
        """
        docs = self.folderReader(self.NewFolderName)
        if docs != False:
            for doc in docs:
                start = doc.find("\\")+1
                end  = len(doc)-4
                print(doc[start:end])
                index = RouteDocument.router().invoke({"question":doc[start:end]})
                print(index.datasource)
                self.loadPDFDoc(doc,index.datasource)
                self.moveFile(doc)

        else:
            return False

    def moveFile(self, doc):
        """
        move file from newFolder to ExistedFile
        
        """
        try:
            start = doc.find("\\")+1
            end  = len(doc)
            os.rename(doc, f"Adoptive_RGA/PDFfolder/existedFile/{doc[start:end]}")
        except:
            st.write(f"file {doc} already exist")
            print(f"file {doc} already exist")
            os.remove(doc)

    def getSpecStore(self):
        """
            return the retriever for Spec vectore store
        """
        emb = OpenAIEmbeddings(model="text-embedding-3-large")
        return FAISS.load_local(self.DBpath,embeddings=emb,index_name="SPECvectorstore",allow_dangerous_deserialization=True).as_retriever()

    def getRegStore(self):
        """
            return the retriever for Reg vectore store
        """
        emb = OpenAIEmbeddings(model="text-embedding-3-large")
        return FAISS.load_local(self.DBpath,embeddings=emb,index_name="REGvectorstore",allow_dangerous_deserialization=True).as_retriever()



    def loadPDFDoc(self, doc, index):   
        """
            load file in the newFolder to local database, 
            if the file already exist in the database, file will not be added again. 
        """
        loader = PDFMinerLoader(doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunk = loader.load_and_split(text_splitter=text_splitter)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(chunk,embedding=embeddings)
        if os.path.exists(f"Adoptive_RGA/faiss_index/{index}.pkl"):
            local_index=FAISS.load_local(self.DBpath,embeddings=embeddings,index_name=f"{index}",allow_dangerous_deserialization=True)
            try:
                local_index.merge_from(vector_store)
                local_index.save_local(self.DBpath, index_name=index)
                print("file added")
            except:
                st.write("file already exist")

        else:
            vector_store.save_local(folder_path=self.DBpath,index_name=index)
            print("file added")



    def folderReader(self,name):
        """
            read the pdf files from user input 
            
            parameter: name: the path of newFolder 

            return: a list of path for each pdf file 
            EX: ["Adoptive_RGA\PDFfolder\existedFile\中華民國無線電頻率分配表.pdf", ...]
        """
        paths = []
        path = f"{name}"
        for filename in os.listdir(path):
            if filename.endswith('.pdf') or filename.endswith('.PDF'):  # Check if the file is a PDF
                pdf_path = os.path.join(path, filename)
                paths.append(pdf_path)
            else:
                f = name+"\\"+filename
                if Path(f).is_dir():
                    print(f"----enter directory {f}----------")
                    for file in os.listdir(f):
                        if file.endswith('.pdf') or file.endswith('.PDF'):  # Check if the file is a PDF
                            pdf_path = os.path.join(f, file)
                            print(f"-----{file}-----")
                            paths.append(pdf_path)

        if paths==[]:
            return False
        else:
            return paths
        

class RouteDocument(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["SPECvectorstore", "REGvectorstore"] = Field(
        ...,
        description="給定一個檔案名稱，當檔案名稱等於無線電頻率供應計畫或是中華民國無線電頻率分配表時，選擇將其路由至special_vectorstore，或當檔案名稱不包含無線電頻率供應計畫或中華民國無線電頻率分配表時，將其路由至regular_vectorstore。",
    )
    # LLM with function call


    def router():
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm_router = llm.with_structured_output(RouteDocument)

        # Prompt
        system = """您是一位專精於將檔案名稱路由至不同類別向量存儲的專家。
                SPECvectorstore只存儲檔案名稱含有無線電頻率供的檔案，其餘檔案通通儲存至REGvectorstore。
                將檔案名稱含有無線電頻率，使用SPECvectorstore。否則，使用REGvectorstore。"""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | structured_llm_router
        return question_router