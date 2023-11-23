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

    def test_invalid_delete(self):
        self.assertFalse(self.graph.is_valid_delete_source_vertex(6))


if __name__ == "__main__":
    unittest.main()