from src.polygraph import PolyGraph
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 4, 2],
        [1, 3, 2, 4]
    ]
    graph = PolyGraph.from_face_loops(face_loops)
    return graph


class TestDeleteVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph._delete_interior_vertex(2)
        self.hidx = [0, 1, 3, 4, 5, 7]

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 4)
        self.assertEqual(self.graph.number_of_halfedges(), 6)
        self.assertEqual(self.graph.number_of_faces(), 2)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 4)
        self.assertEqual(self.graph.number_of_nodes(), 16)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("next"), 6)
        self.assertEqual(self.graph.number_of_edges_of_type("previous"), 6)
        self.assertEqual(self.graph.number_of_edges_of_type("twin"), 10)
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 20)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 20)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 12)

        self.assertEqual(self.graph.number_of_edges(), 74)

    def test_next_edges(self):
        next_idx = [1, 3, 0, 5, 7, 4]
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.next_halfedge(idx) == (ne, htag) for idx, ne in zip(self.hidx, next_idx))
        )

    def test_previous_edges(self):
        next_idx = [1, 3, 0, 5, 7, 4]
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.previous_halfedge(ne) == (idx, htag) for (idx, ne) in zip(self.hidx, next_idx))
        )

    def test_twin_edges(self):
        b = self.graph.boundary_tag
        h = self.graph.halfedge_tag
        twin_idx = [0, 7, 1, 2, 3, 1]
        twin_tags = [b, h, b, b, b, h]

        self.assertTrue(
            all(self.graph.twin_halfedge(idx) == (tidx, tag) for idx, tidx, tag in zip(self.hidx, twin_idx, twin_tags))
        )
        self.assertTrue(all(
            self.graph.twin_halfedge((tidx, tag)) == (idx, h) for idx, tidx, tag in
            zip(self.hidx, twin_idx, twin_tags)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        source_vertices = [0, 1, 2, 1, 3, 2]

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, source_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, source_vertices))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        target_vertices = [1, 2, 0, 3, 2, 1]

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, target_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 0, 3, 2]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 2, 1, 3]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        face_idx = 3 * [0] + 3 * [1]

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in zip(self.hidx, face_idx))
        )
        ftag = self.graph.face_tag
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (idx, htag)) for idx, fidx in zip(self.hidx, face_idx))
        )

    def test_vertex_degree(self):
        degrees = [2, 3, 3, 2]

        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in zip(range(4), degrees))
        )

    def test_face_degree(self):
        self.assertTrue(all(self.graph.face_degree(fidx) == 3 for fidx in range(2)))


if __name__ == "__main__":
    unittest.main()
