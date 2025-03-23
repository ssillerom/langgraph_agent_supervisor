from src.graph import build_graph

from dotenv import load_dotenv

load_dotenv()


app = build_graph()

for s in app.stream(
    {"messages": [("user", "What's the square root of 42?")]}, subgraphs=True
):
    print(s)
    print("----")