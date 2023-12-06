from src.polygraph import PolyGraph
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4],
        [0, 4, 3]
    ]
    graph = PolyGraph.from_face_loops(face_loops)
    return graph


class TestDeleteVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        vidx = self.graph.source_vertex(4, tag=False)
        self.graph.user_defined_vertices.remove(vidx)

    def test_invalid_delete(self):
        self.assertFalse(self.graph.is_valid_delete_source_vertex(6))
        self.assertFalse(self.graph.is_valid_delete_source_vertex(4))


class TestInvalidDeleteSourceTarget(unittest.TestCase):
    def setUp(self) -> None:
        loops = [
            [0, 1, 5, 6, 10, 5, 1, 2, 3, 4],
            [5, 10, 6, 5, 9, 8, 7],
            [5, 7, 8, 9]
        ]
        graph = PolyGraph.from_face_loops(loops)
        graph.user_defined_vertices.remove(6)
        graph.user_defined_vertices.remove(10)
        graph.delete_source_vertex(4)
        self.graph = graph

    def test_invalid_delete(self):
        self.assertFalse(self.graph.is_valid_delete_source_vertex(3))
        self.assertFalse(self.graph.is_valid_delete_source_vertex(12))



if __name__ == "__main__":
    unittest.main()
