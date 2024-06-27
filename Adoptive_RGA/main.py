"""
LAW-AND-REGULATION-HELPER
created by Lawrence Chen 


"""

import streamlit as st

import os
import openai
import dotenv
from dotenv import load_dotenv, find_dotenv
from  PdfMinerFileReader import FileReader
import pydantic.v1.error_wrappers

import threading 

from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime import get_instance
import datetime
from pprint import pprint
from Build_graph import BuildGraph


# setting the api key
# the Langchain_api_key is for traceing LLM output (for developer only)
load_dotenv(find_dotenv())
os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.getenv("LANGCHAIN_API_KEY")







def user_input(user_question):
    """
    the user input function reads the user inputs and pass the input to BuildGraph where the agent will take place
    
    Parameter: user_question: the question user asked
    """
    file_exists = os.path.exists("Adoptive_RGA/faiss_index/REGvectorstore.pkl") and os.path.exists("Adoptive_RGA/faiss_index/SPECvectorstore.pkl")
    if not file_exists:
        st.write("沒有檔案!")
        return False

    inputs = {
        "question": f"{user_question}"
    }
    buildGraph = BuildGraph()
    try:
        value = buildGraph.build(inputs)
    except pydantic.v1.error_wrappers.ValidationError:
        value = 'Please ender your openAI api key'
        return value
    except ValueError:
        value = buildGraph.compile(inputs)
    except openai.RateLimitError:
        value = "the aip key is invalid or you have exceed the limit"
    finally:
        pprint("\n---\n")
        return value
    
def extractMetadata(docs:list):
    source = []
    for doc in docs:
        source.append(doc.metadata['source'])
    return source 





def main():

    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI💁")
    
    with st.form('my_form'):
        user_question = st.text_input("Ask a Question from the PDF Files") 
        submitted = st.form_submit_button("Submit")
    with st.sidebar:
        api_key=st.text_input("請輸入你的open AI API KEY")
        os.environ["OPENAI_API_KEY"] = api_key
        dotenv.set_key(find_dotenv(), "OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
        st.title("新增資料")
        pdf_doc = st.file_uploader("上傳新增的pdf檔", accept_multiple_files=True)
        if st.button("上傳"):
            with st.spinner("processing ....."):
                if pdf_doc != None:
                    for doc in pdf_doc:
                        with open(f'Adoptive_RGA/PDFfolder/newData/{doc.name}', 'wb') as f:
                            # src = doc._file_urls.upload_url + "/doc.name"
                            # shutil.copyfile(src,f'demo/PDFfolder/regData/{doc.name}')
                            f.write(doc.read())
                success = FileReader().addFileToVectorStore()
                if success ==False:
                    st.write("檔案上傳失敗")
                st.success("完成")
        if st.button("清除全部記憶"):
            removeExistFile()
    if submitted:
        response = user_input(user_question)
        if response == "the aip key is invalid or you have exceed the limit":
            st.write("the aip key is invalid or you have used up all your open ai credit")
        elif response == 'Please ender your openAI api key':
            st.write("'Please ender your openAI api key'")
        elif response == False:
            pass
        else:
            st.write("Reply: ", response["generation"])
            if response["documents"]!=[]:
                source = extractMetadata(response["documents"])
                st.write("source:",source)


def removeExistFile():
    
    try:
        os.remove("Adoptive_RGA/faiss_index/REGvectorstore.faiss")
        os.remove("Adoptive_RGA/faiss_index/REGvectorstore.pkl")
    except:
        print("Adoptive_RGA/faiss_index/REGvectorstore.faiss already deleted")
    finally:
        try:
            os.remove("Adoptive_RGA/faiss_index/SPECvectorstore.faiss")
            os.remove("Adoptive_RGA/faiss_index/SPECvectorstore.pkl")
        except:
            print("Adoptive_RGA/faiss_index/SPECvectorstore.faiss already deleted")
        finally:
            docs = FileReader().folderReader("Adoptive_RGA/PDFfolder/existedFile")
            try:
                for doc in docs:
                    os.remove(doc)
            except:
                print("PDF file already deleted")




def start_beating(user_id):
    thread = threading.Timer(interval=2, function=start_beating, args=(user_id,) )

    # insert context to the current thread, needed for 
    # getting session specific attributes like st.session_state

    add_script_run_ctx(thread)

     # context is required to get session_id of the calling
     # thread (which would be the script thread) 
    ctx = get_script_run_ctx()     

    runtime = get_instance()     # this is the main runtime, contains all the sessions
    if runtime.is_active_session(session_id=ctx.session_id):
        # Session is running
        thread.start()
    else:
        os._exit(0)



if __name__ == "__main__":
    user_id = get_script_run_ctx().session_id
    main()
    start_beating(user_id)