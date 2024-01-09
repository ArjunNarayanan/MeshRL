from src.tiler import Tiler

graph = Tiler.from_face_loops([[0, 1, 2, 3]])
graph.insert_half_edge(0, 1)
graph.insert_vertex(4)
graph.insert_vertex(6)
graph.insert_half_edge(0, 2)
graph.delete_half_edge(4)
# graph.insert_half_edge(9, 2)
