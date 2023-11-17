from src.polygraph import PolyGraph
import unittest


def initialize_graph():
    face_loops = [
        list(range(14))
    ]
    graph = PolyGraph.from_face_loops(face_loops)
    return graph


class TestSplitPolygon(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.insert_halfedge(0, 2)
        self.graph.insert_halfedge(15, 2)
        self.graph.insert_halfedge(17, 2)
        self.graph.delete_halfedge(15)
        self.graph.delete_halfedge(16)
        self.hidx = list(range(14)) + [18, 19]

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 14)
        self.assertEqual(self.graph.number_of_halfedges(), 16)
        self.assertEqual(self.graph.number_of_faces(), 2)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 14)
        self.assertEqual(self.graph.number_of_nodes(), 46)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("next"), 16)
        self.assertEqual(self.graph.number_of_edges_of_type("previous"), 16)
        self.assertEqual(self.graph.number_of_edges_of_type("twin"), 30)
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 60)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 60)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 32)

        self.assertEqual(self.graph.number_of_edges(), 214)

    def test_next_edges(self):
        next_idx = [1, 2, 3, 4, 5, 6, 18, 8, 9, 10, 11, 12, 13, 19, 0, 7]
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.next_halfedge(idx) == (ne, htag) for idx, ne in zip(self.hidx, next_idx))
        )

    def test_previous_edges(self):
        next_idx = [1, 2, 3, 4, 5, 6, 18, 8, 9, 10, 11, 12, 13, 19, 0, 7]
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.previous_halfedge(ne) == (idx, htag) for (idx, ne) in zip(self.hidx, next_idx))
        )

    def test_twin_edges(self):
        b = self.graph.boundary_tag
        h = self.graph.halfedge_tag
        twin_idx = list(range(14)) + [19, 18]
        twin_tags = 14 * [b] + 2 * [h]

        self.assertTrue(
            all(self.graph.twin_halfedge(idx) == (tidx, tag) for idx, tidx, tag in zip(self.hidx, twin_idx, twin_tags))
        )
        self.assertTrue(all(
            self.graph.twin_halfedge((tidx, tag)) == (idx, h) for idx, tidx, tag in
            zip(self.hidx, twin_idx, twin_tags)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        source_vertices = list(range(14)) + [7, 0]

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, source_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, source_vertices))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        target_vertices = list(range(1, 14)) + [0, 0, 7]

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, target_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = list(range(1, 14)) + [0]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = range(14)
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        face_idx = 7 * [2] + 7 * [0] + [2, 0]

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in zip(self.hidx, face_idx))
        )
        ftag = self.graph.face_tag
        htag = self.graph.halfedge_tag
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (idx, htag)) for idx, fidx in zip(self.hidx, face_idx))
        )

    def test_vertex_degree(self):
        degrees = [3] + 6 * [2] + [3] + 6 * [2]

        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in enumerate(degrees))
        )

    def test_face_degree(self):
        self.assertTrue(
            all(self.graph.face_degree(fidx) == 8 for fidx in [0, 2])
        )


if __name__ == "__main__":
    unittest.main()
