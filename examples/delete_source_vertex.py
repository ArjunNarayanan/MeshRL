from src.polygraph import PolyGraph

if __name__ == "__main__":
    loops = [
        [0, 1, 5, 6, 10, 5, 1, 2, 3, 4],
        [5, 10, 6, 5, 9, 8, 7],
        [5, 7, 8, 9]
    ]
    graph = PolyGraph.from_face_loops(loops)
    graph.user_defined_vertices.remove(6)
    graph.user_defined_vertices.remove(10)
    graph.delete_source_vertex(4)
    print("Valid : ", graph.is_valid_delete_source_vertex(10))
