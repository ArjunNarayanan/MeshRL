from src.tiler import Tiler


if __name__=="__main__":
    graph = Tiler.from_face_loops(
        [
            [0, 1, 2, 1, 3, 4, 5]
        ]
    )
    graph.insert_vertex(1)
