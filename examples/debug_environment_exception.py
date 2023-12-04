import pickle
from src.polygraph import PolyGraph
from src.render import Renderer

vertex_coords = [
    [0, 0],
    [1, 0],
    [2, 0],
    [2, 3],
    [0, 3],
    [1, 1],
    [1.5, 1],
    [1.5, 2],
    [0.5, 2],
    [0.5, 1]
]
vertex_coords = dict(zip(range(10), vertex_coords))
face_loop = [0, 1, 5, 9, 8, 7, 6, 5, 1, 2, 3, 4]
graph = PolyGraph.from_face_loops([face_loop], vertex_coordinates=vertex_coords)
renderer = Renderer(graph, graph.vertex_coordinates)
renderer.plot()

# filename = "experiments/hex_env_with_insert/incremental-reward/except_env_0.pkl"
# with open(filename, "rb") as f:
#     data = pickle.load(f)
#
# graph = data["graph"]
# renderer = Renderer(graph, graph.vertex_coordinates, vertex_size=10)
# renderer.plot()
#
# actions = data["actions"]
#
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
