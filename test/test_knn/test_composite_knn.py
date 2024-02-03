from src.polygraph import PolyGraph
import unittest


def initialize_graph():
    face_loops = [
        [11, 12, 14, 15, 0, 1],
        [3, 10, 11, 1, 2],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [11, 10, 9, 12],
        [12, 9, 13]
    ]
    graph = PolyGraph.from_face_loops(face_loops)
    return graph


class TestKNN(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_knn_15_10(self):
        neighbors = self.graph.knn_halfedges(15, 10)
        self.assertEqual(len(neighbors), 10)
        test_neighbors = [15, 16, 14, 17, 13, 18, 20, 12, 11, 6]
        test_neighbors = [(h, self.graph.halfedge_tag) for h in test_neighbors]
        self.assertEqual(len(test_neighbors), 10)
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )

    def test_knn_19_12(self):
        neighbors = self.graph.knn_halfedges(19, 12)
        self.assertEqual(len(neighbors), 12)
        test_neighbors = [19, 20, 22, 7, 21, 17, 0, 8, 6, 23, 18, 16]
        test_neighbors = [(t, self.graph.halfedge_tag) for t in test_neighbors]
        self.assertEqual(len(test_neighbors), 12)
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )


if __name__ == "__main__":
    unittest.main()

