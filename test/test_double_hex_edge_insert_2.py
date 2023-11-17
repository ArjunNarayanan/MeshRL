from src.polygraph import PolyGraph
import unittest


def initialize_double_hex_graph():
    face_loops = [
        [0, 1, 2, 7, 8, 9],
        [2, 3, 4, 5, 6, 7]
    ]

    graph = PolyGraph.from_face_loops(face_loops)
    graph.insert_halfedge(2, 3)

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
        next_edges = [1, 13, 3, 4, 5, 12, 7, 8, 9, 10, 11, 6, 2, 0]
        htag = self.graph.halfedge_tag

        self.assertTrue(all(self.graph.next_halfedge(hidx) == (n, htag) for (hidx, n) in zip(range(14), next_edges)))

    def test_previous_edges(self):
        next_edges = [1, 13, 3, 4, 5, 12, 7, 8, 9, 10, 11, 6, 2, 0]
        htag = self.graph.halfedge_tag

        self.assertTrue(
            all(self.graph.previous_halfedge(n) == (hidx, htag) for (n, hidx) in zip(next_edges, range(14)))
        )

    def test_source_vertex(self):
        source_verts = [0, 1, 2, 7, 8, 9, 2, 3, 4, 5, 6, 7, 0, 2]
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag

        self.assertTrue(
            all(self.graph.source_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(14), source_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in zip(range(14), source_verts))
        )

    def test_target_vertex(self):
        target_verts = [1, 2, 7, 8, 9, 0, 3, 4, 5, 6, 7, 2, 2, 0]
        vtag = self.graph.vertex_tag
        htag = self.graph.halfedge_tag

        self.assertTrue(
            all(self.graph.target_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(14), target_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in zip(range(14), target_verts))
        )

    def test_face_edges(self):
        face_ids = [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 0]
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

    def test_vertex_degree(self):
        vertex_degrees = [3, 2, 4, 2, 2, 2, 2, 3, 2, 2]
        num_vertices = self.graph.number_of_vertices()
        self.assertEqual(num_vertices, len(vertex_degrees))
        self.assertTrue(all(self.graph.vertex_degree(vidx) == vd for (vidx, vd) in
                            zip(range(self.graph.number_of_vertices()), vertex_degrees)))

    def test_face_degree(self):
        face_degrees = [3, 6, 5]
        num_faces = self.graph.number_of_faces()
        self.assertEqual(len(face_degrees), num_faces)
        self.assertTrue(all(self.graph.face_degree(fidx) == fd for (fidx, fd) in zip(range(num_faces), face_degrees)))

    def test_valid_edge_insert(self):
        hidx = (2, self.graph.halfedge_tag)
        with self.assertRaises(AssertionError):
            self.graph.insert_halfedge(hidx, 3)
        self.assertFalse(self.graph.is_valid_edge_insert(hidx, 3))
        self.assertTrue(self.graph.is_valid_edge_insert(hidx, 2))

        hidx = (6, self.graph.halfedge_tag)
        self.assertTrue(self.graph.is_valid_edge_insert(hidx, 1))
        self.assertTrue(self.graph.is_valid_edge_insert(hidx, 2))
        self.assertTrue(self.graph.is_valid_edge_insert(hidx, 3))
        self.assertFalse(self.graph.is_valid_edge_insert(hidx, 4))


if __name__ == "__main__":
    unittest.main()
