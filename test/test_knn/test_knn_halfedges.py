from src.polygraph import PolyGraph
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    graph = PolyGraph.from_face_loops(face_loops)
    return graph


class TestKNN(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_knn_3(self):
        neighbors = self.graph.knn_halfedges(0, 3)
        h = self.graph.halfedge_tag
        test_neighbors = [(0, h), (1, h), (5, h)]
        self.assertEqual(len(neighbors), 3)
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )

    def test_knn_10(self):
        neighbors = self.graph.knn_halfedges(2, 10)
        self.assertEqual(len(neighbors), 6)
        h = self.graph.halfedge_tag
        test_neighbors = [2, 3, 1, 4, 0, 5]
        test_neighbors = [(n, h) for n in test_neighbors]
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )

    def test_knn_6(self):
        neighbors = self.graph.knn_halfedges(2, 6)
        self.assertEqual(len(neighbors), 6)
        h = self.graph.halfedge_tag
        test_neighbors = [2, 3, 1, 4, 0, 5]
        test_neighbors = [(n, h) for n in test_neighbors]
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )
