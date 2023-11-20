from src.polygraph import PolyGraph
import unittest
import numpy as np


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    coords = generate_coordinates()
    graph = PolyGraph.from_face_loops(face_loops, coords)
    return graph


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(6), coords))
    return coords


class TestDeleteEdge(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.hidx = range(18)

        self.inserted_face1 = self.graph.insert_halfedge(0, 2)
        self.inserted_vertex1 = self.graph.insert_vertex(7)
        self.inserted_face2 = self.graph.insert_halfedge(8, 1)
        self.inserted_face3 = self.graph.insert_halfedge(5, 1)
        self.inserted_face4 = self.graph.insert_halfedge(6, 1)
        self.inserted_face5 = self.graph.insert_halfedge(15, 1)

    def test_new_indices(self):
        ftag = self.graph.face_tag
        vtag = self.graph.vertex_tag
        self.assertEqual(self.inserted_face1, (1, ftag))
        self.assertEqual(self.inserted_vertex1, (6, vtag))
        self.assertEqual(self.inserted_face2, (2, ftag))
        self.assertEqual(self.inserted_face3, (3, ftag))
        self.assertEqual(self.inserted_face4, (4, ftag))
        self.assertEqual(self.inserted_face5, (5, ftag))

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 7)
        self.assertEqual(self.graph.number_of_halfedges(), 18)
        self.assertEqual(self.graph.number_of_faces(), 6)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 6)
        self.assertEqual(self.graph.number_of_nodes(), 37)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("next"), 18)
        self.assertEqual(self.graph.number_of_edges_of_type("previous"), 18)
        self.assertEqual(self.graph.number_of_edges_of_type("twin"), 24)
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 48)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 48)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 36)

        self.assertEqual(self.graph.number_of_edges(), 192)

    def test_next_edges(self):
        next_idx = [14, 16, 9, 10, 13, 7, 0, 12, 3, 17, 8, 4, 5, 11, 6, 1, 15, 2]
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.next_halfedge(idx) == (ne, htag) for idx, ne in enumerate(next_idx))
        )

    def test_previous_edges(self):
        next_idx = [14, 16, 9, 10, 13, 7, 0, 12, 3, 17, 8, 4, 5, 11, 6, 1, 15, 2]
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.previous_halfedge(ne) == (idx, htag) for (idx, ne) in enumerate(next_idx))
        )

    def test_twin_edges(self):
        b = self.graph.boundary_tag
        h = self.graph.halfedge_tag
        twin_idx = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]
        twin_tags = 6 * [b] + 12 * [h]

        self.assertTrue(
            all(self.graph.twin_halfedge(idx) == (tidx, tag) for idx, tidx, tag in zip(self.hidx, twin_idx, twin_tags))
        )
        self.assertTrue(all(
            self.graph.twin_halfedge((tidx, tag)) == (idx, h) for idx, tidx, tag in
            zip(self.hidx, twin_idx, twin_tags)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        source_vertices = [0, 1, 2, 3, 4, 5, 6, 0, 6, 3, 4, 6, 6, 5, 1, 6, 2, 6]

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, source_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, source_vertices))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        target_vertices = [1, 2, 3, 4, 5, 0, 0, 6, 3, 6, 6, 4, 5, 6, 6, 1, 6, 2]

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, target_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 2, 3, 4, 5, 0]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = range(6)
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        face_idx = [4, 5, 1, 2, 0, 3, 4, 3, 2, 1, 2, 0, 3, 0, 4, 5, 5, 1]

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in zip(self.hidx, face_idx))
        )
        ftag = self.graph.face_tag
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (idx, htag)) for idx, fidx in zip(self.hidx, face_idx))
        )

    def test_vertex_degree(self):
        degrees = 6 * [3] + [6]

        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in zip(range(7), degrees))
        )

    def test_face_degree(self):
        self.assertTrue(
            all(self.graph.face_degree(fidx) == 3 for fidx in range(6))
        )


if __name__ == "__main__":
    unittest.main()