import streamlit as st

import os

from dotenv import load_dotenv

import fileReader
import PdfMinerFileReader


from pprint import pprint
from Build_graph import BuildGraph
load_dotenv()
os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.getenv("LANGCHAIN_API_KEY")









def user_input(user_question):
    ret = PdfMinerFileReader.FileReader()
    inputs = {
        "question": f"{user_question}"
    }
    buildGraph = BuildGraph(ret)
    for output in buildGraph.build(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
    return value



def main():
    # fileReadInstance = PdfMinerFileReader.FileReader()
    memory ={}
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question)
        st.write("Reply: ", response["generation"])





if __name__ == "__main__":
    main()