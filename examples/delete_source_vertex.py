from src.polygraph import PolyGraph

face_loops = [
    [0, 1, 2, 3, 4],
    [0, 4, 3]
]
graph = PolyGraph.from_face_loops(face_loops)
graph.user_defined_vertices.remove(4)
graph.delete_source_vertex(4)

halfedges = [0, 1, 2, 3, 6, 7]
for hidx in halfedges:
    print("hidx: ", hidx)
    print("next: ", graph.next_halfedge(hidx))
    print("prev: ", graph.previous_halfedge(hidx))
    print("twin: ", graph.twin_halfedge(hidx))
    print("src : ", graph.source_vertex(hidx))
    print("dst : ", graph.target_vertex(hidx))
    print("\n")
