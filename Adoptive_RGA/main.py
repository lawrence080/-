import streamlit as st

import os

from dotenv import load_dotenv
import json
import fileReader
import PdfMinerFileReader


from pprint import pprint
from Build_graph import BuildGraph
load_dotenv()
os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.getenv("LANGCHAIN_API_KEY")







def user_input(user_question):
    # ret = PdfMinerFileReader.FileReader()
    inputs = {
        "question": f"{user_question}"
    }
    buildGraph = BuildGraph()
    try:
        value = buildGraph.build(inputs)
        # for output in buildGraph.build(inputs):
        #     for key, value in output.items():
        #         # Node
        #         pprint(f"Node '{key}':")
        #         # Optional: print full state at each node
        #         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        print(value)
        pprint("\n---\n")
    except ValueError:
        value = buildGraph.compile(inputs)
        # for output in buildGraph.compile(inputs):
        #     for key, value in output.items():
        #         # Node
        #         pprint(f"Node '{key}':")
        #         # Optional: print full state at each node
        #         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        #     pprint("\n---\n")
        print(value)
        pprint("\n---\n")
    finally:
        return value
    
def extractMetadata(docs:list):
    source = []
    for doc in docs:
        source.append(doc.metadata['source'])
    return source 





def main():
    #fileReadInstance = PdfMinerFileReader.FileReader()
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI💁")
    
    with st.form('my_form'):
        user_question = st.text_input("Ask a Question from the PDF Files") 
        submitted = st.form_submit_button("Submit")
    with st.sidebar:
        st.title("新增資料")
        pdf_doc = st.file_uploader("上傳新增的pdf檔")
        if st.button("上傳"):
            with st.spinner("processing ....."):
                st.success("完成")
    if submitted:
        response = user_input(user_question)
        # print(response["documents"][0].metadata)
        source = extractMetadata(response["documents"])
        st.write("Reply: ", response["generation"])
        st.write("source:",source)





if __name__ == "__main__":
    # buildGraph = BuildGraph(ret)
    main()