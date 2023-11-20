from src.polygraph import PolyGraph
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    graph = PolyGraph.from_face_loops(face_loops)
    graph.insert_halfedge(0, 2)
    graph.insert_vertex(7)
    graph.insert_halfedge(8, 1)
    graph.insert_halfedge(5, 1)
    graph.insert_halfedge(6, 1)
    graph.insert_halfedge(15, 1)
    return graph


class TestKNN(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_knn_13_8(self):
        neighbors = self.graph.knn_halfedges(13, 8)
        htag = self.graph.halfedge_tag
        test_neighbors = [13, 11, 4, 12, 10, 5, 7, 8]
        test_neighbors = [(n, htag) for n in test_neighbors]

        self.assertEqual(len(neighbors), 8)
        self.assertTrue(
            all(h == t for h, t in zip(neighbors, test_neighbors))
        )

    def test_knn_2_15(self):
        neighbors = self.graph.knn_halfedges(2, 15)
        htag = self.graph.halfedge_tag
        self.assertEqual(len(neighbors), 15)
        test_neighbors = [2, 9, 17, 8, 16, 3, 10, 15, 1, 11, 14, 4, 13, 6, 0]
        test_neighbors = [(n, htag) for n in test_neighbors]
        self.assertEqual(len(test_neighbors), 15)
        self.assertTrue(
            all(t == n for t, n in zip(test_neighbors, neighbors))
        )

    def test_knn_6_18(self):
        neighbors = self.graph.knn_halfedges(6, 18)
        self.assertEqual(len(neighbors), 18)
        htag = self.graph.halfedge_tag
        test_neighbors = [6, 0, 14, 7, 15, 12, 5, 1, 16, 13, 17, 11, 4, 2, 9, 10, 8, 3]
        test_neighbors = [(n, htag) for n in test_neighbors]
        self.assertEqual(len(test_neighbors), 18)
        self.assertTrue(
            all(t == n for t, n in zip(test_neighbors, neighbors))
        )
