from pprint import pprint
import Build_graph
import os
# Run

from dotenv import load_dotenv

load_dotenv()
os.getenv("TAVILY_API_KEY")

userinput = input("請輸入問題")
inputs = {
    "question": f"{userinput}"
}
for output in Build_graph.app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])