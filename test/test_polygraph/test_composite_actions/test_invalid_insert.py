from src.polygraph import PolyGraph
import unittest


class TestInvalidInsert(unittest.TestCase):
    def setUp(self) -> None:
        graph = PolyGraph.from_face_loops([[0, 1, 2, 3]])
        graph.insert_halfedge(0, 1)
        graph.insert_vertex(4)
        graph.insert_vertex(6)
        graph.insert_halfedge(0, 2)
        graph.delete_halfedge(4)
        self.graph = graph

    def test_invalid_action(self):
        self.assertFalse(self.graph.is_valid_edge_insert(9, 2))


if __name__ == "__main__":
    unittest.main()
