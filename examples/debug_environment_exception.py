import pickle
from src.polygraph import PolyGraph
from src.render import Renderer



filename = "experiments/regular-polygon/poly-20/debug-log/except-env-1.pkl"
with open(filename, "rb") as f:
    data = pickle.load(f)

graph = data["graph"]
renderer = Renderer(graph, graph.vertex_coordinates, vertex_size=10)
renderer.plot()

actions = data["actions"]

# graph.insert_halfedge(2,1)
# renderer.plot()
#
# graph.insert_halfedge(7,2)
# renderer.plot()
#
# graph.insert_vertex(8)
# renderer.plot()
#
# graph.insert_vertex(8)
# renderer.plot()
#
# graph.insert_halfedge(5, 2)
# renderer.plot()
#
# graph.insert_vertex(13)
# renderer.plot()
#
# graph.insert_vertex(5)
# renderer.plot()
#
# graph.insert_halfedge(10, 1)
# renderer.plot()
#
# graph.insert_vertex(12)
# renderer.plot()
#
# graph.insert_halfedge(13,1)
# renderer.plot()
#
# graph.delete_halfedge(21)
# renderer.plot()
#
# graph.delete_halfedge(8)
# renderer.plot()
#
# # graph.insert_vertex(0)
# # graph.insert_vertex(7)
# # graph.insert_halfedge(10, 2)
# # graph.insert_vertex(6)
# # graph.insert_halfedge(13,2)
# # graph.delete_halfedge(13)
# # graph.insert_vertex(3)
# # graph.insert_halfedge(15, 2)
#
#
