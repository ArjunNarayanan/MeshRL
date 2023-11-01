from src.polygraph import PolyGraph
import unittest


def initialize_double_hex_graph():
    face_loops = [
        [0, 1, 2, 7, 8, 9],
        [2, 3, 4, 5, 6, 7]
    ]

    graph = PolyGraph(face_loops)
    graph.insert_edge(0, 1)

    return graph


class TestInsertHalfedge(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_double_hex_graph()

    def test_num_graph_nodes(self):
        num_halfedge = self.graph.number_of_halfedges()
        self.assertEqual(num_halfedge, 14)

        num_verts = self.graph.number_of_vertices()
        self.assertEqual(num_verts, 10)

        num_faces = self.graph.number_of_faces()
        self.assertEqual(num_faces, 3)

        num_boundary = self.graph._number_of_nodes("boundary")
        self.assertEqual(num_boundary, 10)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 37)

    def test_num_edges(self):
        next_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "next"]
        self.assertEqual(len(next_edges), 14)

        previous_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "previous"]
        self.assertEqual(len(previous_edges), 14)

        source_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "source"]
        self.assertEqual(len(source_edges), 48)

        target_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "target"]
        self.assertEqual(len(target_edges), 48)

        face_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "face"]
        self.assertEqual(len(face_edges), 28)

        num_twin = sum(1 for src, dst, data in self.graph.edges(data=True) if data.get("type") == "twin")
        self.assertEqual(num_twin, 24)

        self.assertEqual(len(self.graph.edges()), 176)

    def test_next_edges(self):
        next_edges = [1, 12, 3, 4, 5, 13, 7, 8, 9, 10, 11, 6, 0, 2]
        htag = self.graph.halfedge_tag

        self.assertTrue(all(self.graph.next_halfedge(hidx) == (n, htag) for (hidx, n) in zip(range(14), next_edges)))

    def test_previous_edges(self):
        next_edges = [1, 12, 3, 4, 5, 13, 7, 8, 9, 10, 11, 6, 0, 2]
        htag = self.graph.halfedge_tag

        self.assertTrue(
            all(self.graph.previous_halfedge(n) == (hidx, htag) for (n, hidx) in zip(next_edges, range(14)))
        )

    def test_source_vertex(self):
        source_verts = [0, 1, 2, 7, 8, 9, 2, 3, 4, 5, 6, 7, 2, 0]
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag

        self.assertTrue(
            all(self.graph.source_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(14), source_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in zip(range(14), source_verts))
        )

    def test_target_vertex(self):
        target_verts = [1, 2, 7, 8, 9, 0, 3, 4, 5, 6, 7, 2, 0, 2]
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag

        self.assertTrue(
            all(self.graph.target_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(14), target_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in zip(range(14), target_verts))
        )

    def test_boundary_source(self):
        source_verts = [1, 2, 8, 9, 0, 3, 4, 5, 6, 7]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(all(
            self.graph.source_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in zip(range(10), source_verts)))

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in zip(range(10), source_verts)))

    def test_boundary_target(self):
        target_verts = [0, 1, 7, 8, 9, 2, 3, 4, 5, 6]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(
                self.graph.target_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in zip(range(10), target_verts)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in zip(range(10), target_verts)
            )
        )

    def test_face_edges(self):
        face_ids = [2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 0]
        ftag = self.graph.face_tag
        htag = self.graph.halfedge_tag

        self.assertTrue(
            all(
                self.graph.face(hidx) == (f, ftag) for (hidx, f) in zip(range(14), face_ids)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((fidx, ftag), (hidx, htag)) for (hidx, fidx) in zip(range(14), face_ids)
            )
        )

    def test_twin_edge(self):
        twin_idx = [0, 1, 11, 2, 3, 4, 5, 6, 7, 8, 9, 2, 13, 12]
        h, b = self.graph.halfedge_tag, self.graph.boundary_tag
        tags = [b, b, h, b, b, b, b, b, b, b, b, h, h, h]

        self.assertTrue(
            all(
                self.graph.twin_halfedge(hidx) == (tidx, tg) for (hidx, tidx, tg) in zip(range(14), twin_idx, tags)
            )
        )

        self.assertTrue(
            all(
                self.graph.twin_halfedge((tidx, tag)) == (hidx, h) for (hidx, tidx, tag) in
                zip(range(14), twin_idx, tags)
            )
        )


# graph = initialize_double_hex_graph()
if __name__ == "__main__":
    unittest.main()