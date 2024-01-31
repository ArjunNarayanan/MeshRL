from envs.regular_polygon_env import RegularPolygonEnv
import unittest


class TestActionSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RegularPolygonEnv(
            6,
            template_size=6,
            max_substeps=5,
            max_steps=1,
            incremental_reward=True,
            logdir=""
        )
        self.env.template_center = (4, self.env.graph.halfedge_tag)
        self.env._build_template()

    def test_vertex_insert(self):
        self.env.step(16)

        graph = self.env.graph
        self.assertEqual(graph.number_of_vertices(), 7)
        self.assertEqual(graph.number_of_faces(), 1)
        self.assertEqual(graph.number_of_halfedges(), 7)

        next_edges = [graph.next_halfedge(idx, tag=False) for idx in range(7)]
        test_next = [1, 2, 3, 6, 5, 0, 4]
        self.assertEqual(next_edges, test_next)

        prev_edges = [graph.previous_halfedge(idx, tag=False) for idx in range(7)]
        test_prev = [5, 0, 1, 2, 6, 4, 3]
        self.assertEqual(prev_edges, test_prev)

        src_vertex = [graph.source_vertex(idx, tag=False) for idx in range(7)]
        test_src = [0, 1, 2, 3, 4, 5, 6]
        self.assertEqual(src_vertex, test_src)

        target_vertex = [graph.target_vertex(idx, tag=False) for idx in range(7)]
        test_target = [1, 2, 3, 6, 5, 0, 4]
        self.assertEqual(target_vertex, test_target)

    def test_edge_insert(self):
        self.env.step(25)

        graph = self.env.graph
        self.assertEqual(graph.number_of_vertices(), 6)
        self.assertEqual(graph.number_of_faces(), 2)
        self.assertEqual(graph.number_of_halfedges(), 8)

        next_edges = [graph.next_halfedge(idx, tag=False) for idx in range(8)]
        test_next = [1, 7, 3, 4, 6, 0, 2, 5]
        self.assertEqual(next_edges, test_next)

        prev_edges = [graph.previous_halfedge(idx, tag=False) for idx in range(8)]
        test_prev = [5, 0, 6, 2, 3, 7, 4, 1]
        self.assertEqual(prev_edges, test_prev)

        twin_edges = [graph.twin_halfedge(idx) for idx in range(8)]
        b = graph.boundary_tag
        h = graph.halfedge_tag
        test_twin = [0, 1, 2, 3, 4, 5, 7, 6]
        tags = 6*[b] + 2*[h]
        test_twin = list(zip(test_twin, tags))
        self.assertEqual(twin_edges, test_twin)

        src_vertex = [graph.source_vertex(idx, tag=False) for idx in range(8)]
        test_src = [0, 1, 2, 3, 4, 5, 5, 2]
        self.assertEqual(src_vertex, test_src)

        target_vertex = [graph.target_vertex(idx, tag=False) for idx in range(8)]
        test_target = [1, 2, 3, 4, 5, 0, 2, 5]
        self.assertEqual(target_vertex, test_target)


if __name__ == "__main__":
    unittest.main()
