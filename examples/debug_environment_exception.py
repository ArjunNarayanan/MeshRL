import pickle
from src.polygraph import PolyGraph

filename = "experiments/hex_env_with_insert/eps-0-05/except_env_0.pkl"
with open(filename, "rb") as f:
    data = pickle.load(f)

graph = data["graph"]
actions = data["actions"]

graph.insert_halfedge(4,3)
graph.insert_vertex(0)
graph.insert_vertex(7)
graph.insert_halfedge(10, 2)
graph.insert_vertex(6)
graph.insert_halfedge(13,2)
graph.delete_halfedge(13)
graph.insert_vertex(3)
graph.insert_halfedge(15, 2)