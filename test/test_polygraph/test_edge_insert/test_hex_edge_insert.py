from src.polygraph import PolyGraph
import unittest


def initialize_hex_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5],
    ]

    graph = PolyGraph.from_face_loops(face_loops)

    return graph


class TestInsertHalfedge(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_hex_graph()
        self.face1 = self.graph.insert_halfedge(0, 2)
        self.face2 = self.graph.insert_halfedge(1, 1)
        self.face3 = self.graph.insert_halfedge(4, 1)

    def test_new_face_idx(self):
        ftag = self.graph.face_tag
        self.assertEqual(self.face1, (1, ftag))
        self.assertEqual(self.face2, (2, ftag))
        self.assertEqual(self.face3, (3, ftag))

    def test_num_graph_nodes(self):
        num_halfedge = self.graph.number_of_halfedges()
        self.assertEqual(num_halfedge, 12)

        num_verts = self.graph.number_of_vertices()
        self.assertEqual(num_verts, 6)

        num_faces = self.graph.number_of_faces()
        self.assertEqual(num_faces, 4)

        num_boundary = self.graph._number_of_nodes("boundary")
        self.assertEqual(num_boundary, 6)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 28)

    def test_num_edges(self):
        next_edges = sum(1 for src, dst, data in self.graph.edges(data=True) if data.get("type") == "next")
        self.assertEqual(next_edges, 12)

        previous_edges = sum(1 for src, dst, data in self.graph.edges(data=True) if data.get("type") == "previous")
        self.assertEqual(previous_edges, 12)

        source_edges = sum(1 for src, dst, data in self.graph.edges(data=True) if data["type"] == "source")
        self.assertEqual(source_edges, 36)

        target_edges = sum(1 for src, dst, data in self.graph.edges(data=True) if data["type"] == "target")
        self.assertEqual(target_edges, 36)

        face_edges = sum(1 for src, dst, data in self.graph.edges(data=True) if data["type"] == "face")
        self.assertEqual(face_edges, 24)

        num_twin = sum(1 for src, dst, data in self.graph.edges(data=True) if data.get("type") == "twin")
        self.assertEqual(num_twin, 18)

        self.assertEqual(len(self.graph.edges()), 138)

    def test_num_next_previous_edges(self):
        def num_edges(hidx, edge_type):
            ne = sum(1 for src, dst, data in self.graph.edges(hidx, data=True) if data.get("type") == edge_type)
            return ne

        htag = self.graph.halfedge_tag
        num_halfedges = self.graph.number_of_halfedges()

        self.assertTrue(all(num_edges((hidx, htag), "next") == 1 for hidx in range(num_halfedges)))
        self.assertTrue(all(num_edges((hidx, htag), "previous") == 1 for hidx in range(num_halfedges)))

    def test_next_edges(self):
        next_edges = [9, 2, 8, 11, 5, 10, 0, 3, 1, 6, 4, 7]
        htag = self.graph.halfedge_tag
        num_halfedges = self.graph.number_of_halfedges()

        self.assertTrue(
            all(self.graph.next_halfedge(hidx) == (n, htag) for (hidx, n) in zip(range(num_halfedges), next_edges))
        )

    def test_previous_edges(self):
        next_edges = [9, 2, 8, 11, 5, 10, 0, 3, 1, 6, 4, 7]
        htag = self.graph.halfedge_tag
        num_halfedges = self.graph.number_of_halfedges()

        self.assertTrue(
            all(self.graph.previous_halfedge(n) == (hidx, htag) for (n, hidx) in zip(next_edges, range(num_halfedges)))
        )

    def test_source_vertex(self):
        source_verts = [0, 1, 2, 3, 4, 5, 3, 0, 3, 1, 0, 4]
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        num_halfedges = self.graph.number_of_halfedges()

        self.assertEqual(num_halfedges, len(source_verts))
        self.assertTrue(
            all(self.graph.source_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(num_halfedges), source_verts))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in
                zip(range(num_halfedges), source_verts))
        )

    def test_target_vertex(self):
        target_verts = [1, 2, 3, 4, 5, 0, 0, 3, 1, 3, 4, 0]
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag
        num_halfedges = self.graph.number_of_halfedges()

        self.assertEqual(num_halfedges, len(target_verts))
        self.assertTrue(
            all(self.graph.target_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(num_halfedges), target_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in
                zip(range(num_halfedges), target_verts))
        )

    def test_boundary_source(self):
        source_verts = [1, 2, 3, 4, 5, 0]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        num_boundary = self.graph._number_of_nodes("boundary")

        self.assertEqual(num_boundary, len(source_verts))
        self.assertTrue(
            all(
                self.graph.source_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in
                zip(range(num_boundary), source_verts)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in
                zip(range(num_boundary), source_verts)
            )
        )

    def test_boundary_target(self):
        target_verts = [0, 1, 2, 3, 4, 5]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        num_boundary = self.graph._number_of_nodes("boundary")

        self.assertEqual(num_boundary, len(target_verts))
        self.assertTrue(
            all(
                self.graph.target_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in
                zip(range(num_boundary), target_verts)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in
                zip(range(num_boundary), target_verts)
            )
        )

    def test_face_edges(self):
        face_ids = [1, 2, 2, 0, 3, 3, 1, 0, 2, 1, 3, 0]
        ftag = self.graph.face_tag
        htag = self.graph.halfedge_tag
        num_halfedges = self.graph.number_of_halfedges()

        self.assertEqual(num_halfedges, len(face_ids))
        self.assertTrue(
            all(
                self.graph.face(hidx) == (f, ftag) for (hidx, f) in zip(range(num_halfedges), face_ids)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((fidx, ftag), (hidx, htag)) for (hidx, fidx) in zip(range(num_halfedges), face_ids)
            )
        )

    def test_twin_edge(self):
        twin_idx = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 11, 10]
        h, b = self.graph.halfedge_tag, self.graph.boundary_tag
        tags = [b, b, b, b, b, b, h, h, h, h, h, h]
        num_halfedges = self.graph.number_of_halfedges()

        self.assertEqual(num_halfedges, len(twin_idx))
        self.assertEqual(num_halfedges, len(tags))

        self.assertTrue(
            all(
                self.graph.twin_halfedge(hidx) == (tidx, tg) for (hidx, tidx, tg) in
                zip(range(num_halfedges), twin_idx, tags)
            )
        )

        self.assertTrue(
            all(
                self.graph.twin_halfedge((tidx, tag)) == (hidx, h) for (hidx, tidx, tag) in
                zip(range(num_halfedges), twin_idx, tags)
            )
        )

    def test_assert_add_new_edge(self):
        for hidx in range(self.graph.number_of_halfedges()):
            with self.assertRaises(AssertionError):
                self.graph.insert_halfedge(hidx, 1)
            with self.assertRaises(AssertionError):
                self.graph.insert_halfedge(hidx, 0)

    def test_vertex_degree(self):
        vertex_degree = [4, 3, 2, 4, 3, 2]
        self.assertTrue(
            all(
                self.graph.vertex_degree(vid) == vd for vid, vd in
                zip(range(self.graph.number_of_vertices()), vertex_degree)
            )
        )

    def test_face_degree(self):
        self.assertTrue(all(self.graph.face_degree(fidx) == 3 for fidx in range(self.graph.number_of_faces())))


# graph = initialize_hex_graph()
if __name__ == "__main__":
    unittest.main()
