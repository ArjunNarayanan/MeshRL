from src.polygraph import PolyGraph
import unittest


def initialize_double_hex_graph():
    face_loops = [
        [0, 1, 2, 7, 8, 9],
        [2, 3, 4, 5, 6, 7]
    ]

    graph = PolyGraph(face_loops)

    return graph


class TestHalfEdgeConnectivity(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_double_hex_graph()

    def test_num_edges(self):
        self.assertEqual(len(self.graph.edges()), 158)

        next_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "next"]
        self.assertEqual(len(next_edges), 12)

        previous_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "previous"]
        self.assertEqual(len(previous_edges), 12)

        source_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "source"]
        self.assertEqual(len(source_edges), 44)

        target_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "target"]
        self.assertEqual(len(target_edges), 44)

        face_edges = [(src, dst) for src, dst, data in self.graph.edges(data=True) if data["type"] == "face"]
        self.assertEqual(len(face_edges), 24)

    def test_num_graph_nodes(self):
        num_halfedge = len([v for v, data in self.graph.nodes(data=True) if data.get("type") == "halfedge"])
        self.assertEqual(num_halfedge, 12)

        num_verts = len([v for v, data in self.graph.nodes(data=True) if data.get("type") == "vertex"])
        self.assertEqual(num_verts, 10)

        num_faces = len([v for v, data in self.graph.nodes(data=True) if data.get("type") == "face"])
        self.assertEqual(num_faces, 2)

        num_boundary = len([v for v in self.graph.nodes() if v[1] == self.graph.boundary_tag])
        self.assertEqual(num_boundary, 10)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 34)

    def test_next_edges(self):
        next_edges = list(range(1, 6)) + [0]
        self.assertTrue(
            all(self.graph.next_halfedge(idx) == (ne, self.graph.halfedge_tag) for idx, ne in zip(range(6), next_edges))
        )
        next_edges = list(range(7, 12)) + [6]
        self.assertTrue(
            all(self.graph.next_halfedge(idx) == (ne, self.graph.halfedge_tag) for idx, ne in
                zip(range(6, 12), next_edges))
        )

    def test_previous_edges(self):
        next_edges = list(range(1, 6)) + [0]
        self.assertTrue(
            all(self.graph.previous_halfedge(ne) == (idx, self.graph.halfedge_tag) for idx, ne in
                zip(range(6), next_edges))
        )
        next_edges = list(range(7, 12)) + [6]
        self.assertTrue(
            all(
                self.graph.previous_halfedge(ne) == (idx, self.graph.halfedge_tag) for idx, ne in
                zip(range(6, 12), next_edges)
            )
        )

    def test_source_vertices(self):
        source_vertices = [0, 1, 2, 7, 8, 9, 2, 3, 4, 5, 6, 7]
        self.assertTrue(
            all(self.graph.source_vertex(h) == (v, self.graph.vertex_tag) for (h, v) in
                zip(range(12), source_vertices))
        )

        vertex_ids = [0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9]
        source_halfedge_ids = [0, 1, 2, 6, 7, 8, 9, 10, 11, 3, 4, 5]
        self.assertTrue(
            all(
                self.graph.has_edge((v, self.graph.vertex_tag), (h, self.graph.halfedge_tag)) for (v, h) in
                zip(vertex_ids, source_halfedge_ids)
            )
        )

    def test_target_vertices(self):
        vertex_ids = [0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9]
        target_halfedge_ids = [5, 0, 1, 11, 6, 7, 8, 9, 10, 2, 3, 4]

        self.assertTrue(all(self.graph.target_vertex(h) == (v, self.graph.vertex_tag) for (h, v) in
                            zip(target_halfedge_ids, vertex_ids)))

        self.assertTrue(
            all(
                self.graph.has_edge((v, self.graph.vertex_tag), (h, self.graph.halfedge_tag)) for (v, h) in
                zip(vertex_ids, target_halfedge_ids)
            )
        )

    def test_face_edges(self):
        face_ids = 6 * [0] + 6 * [1]
        self.assertTrue(
            all(
                self.graph.face(halfedge) == (face, self.graph.face_tag) for halfedge, face in zip(range(12), face_ids)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((f, self.graph.face_tag), (h, self.graph.halfedge_tag))
                for (f, h) in zip(face_ids, range(12))
            )
        )

    def test_twin_edges(self):
        twin_id = [0, 1, 11, 2, 3, 4, 5, 6, 7, 8, 9, 2]
        htag = self.graph.halfedge_tag
        btag = self.graph.boundary_tag
        twin_tags = 2 * [btag] + [htag] + 8 * [btag] + [htag]
        twin_nodes = [(node_id, node_tag) for node_id, node_tag in zip(twin_id, twin_tags)]
        self.assertTrue(
            all(self.graph.twin_halfedge(h) == t for h, t in zip(range(self.graph.num_halfedges), twin_nodes))
        )

        halfedge_id = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertTrue(all(self.graph.twin_halfedge((b, btag)) == (h, htag) for b, h in zip(range(10), halfedge_id)))

    def test_vertex_degree(self):
        vertex_degree = [2, 2, 3, 2, 2, 2, 2, 3, 2, 2]
        self.assertTrue(all(self.graph.vertex_degree(vid) == vd for vid, vd in zip(range(10), vertex_degree)))


# graph = initialize_double_hex_graph()
# graph._add_twin_edges()
# graph._add_twin_source_target_edges()

if __name__ == "__main__":
    unittest.main()
