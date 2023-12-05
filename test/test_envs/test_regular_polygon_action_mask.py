from envs.regular_polygon_env import RegularPolygonEnv
import unittest
import numpy as np


class TestActionMask(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RegularPolygonEnv(
            6,
            template_size=8,
            max_substeps=8,
            max_steps=1,
            incremental_reward=False,
            logdir=""
        )

    def test_valid_action(self):
        test_actions = 6 * [True, True, True, False, True, False] + 12 * [False]
        valid_actions = [self.env.is_valid_action(idx) for idx in range(48)]
        self.assertEqual(test_actions, valid_actions)


class TestSequentialActionMask(unittest.TestCase):
    def setUp(self) -> None:
        env = RegularPolygonEnv(
            6,
            template_size=10,
            max_substeps=5,
            max_steps=1,
            incremental_reward=False,
            logdir="",
        )
        env._step_insert_edge(1, 1)
        env._step_insert_vertex(6)
        env._step_insert_edge(7, 2)
        env._step_delete_edge(6)
        env.template_center = (0, env.graph.halfedge_tag)
        env._build_template()
        self.env = env
        self.hidx = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]

    def test_state(self):
        graph = self.env.graph
        self.assertEqual(graph.number_of_halfedges(), 10)
        self.assertEqual(graph.number_of_faces(), 2)
        self.assertEqual(graph.number_of_vertices(), 7)

        next_edges = [graph.next_halfedge(hidx, tag=False) for hidx in self.hidx]
        test_next = [9, 2, 3, 4, 10, 0, 1, 11, 8, 5]
        self.assertEqual(next_edges, test_next)

        prev_edges = [graph.previous_halfedge(hidx, tag=False) for hidx in self.hidx]
        test_prev = [5, 8, 1, 2, 3, 11, 10, 0, 4, 9]
        self.assertEqual(prev_edges, test_prev)

        twin_edges = [graph.twin_halfedge(hidx) for hidx in self.hidx]
        h = graph.halfedge_tag
        b = graph.boundary_tag
        test_twin = [0, 1, 2, 3, 4, 5, 9, 8, 11, 10]
        test_tags = 6 * [b] + 4 * [h]
        test_twin = list(zip(test_twin, test_tags))
        self.assertEqual(twin_edges, test_twin)

    def test_valid_actions(self):
        t = True
        f = False

        test_col0 = [t, f, f, f, t, f]
        col0 = [self.env.is_valid_action(idx) for idx in range(6)]
        self.assertEqual(col0, test_col0)

        test_col1 = [t, f, f, f, t, f]
        col1 = [self.env.is_valid_action(idx) for idx in range(6, 12)]
        self.assertEqual(col1, test_col1)

        test_col2 = [t, f, f, f, t, f]
        col2 = [self.env.is_valid_action(idx) for idx in range(12, 18)]
        self.assertEqual(col2, test_col2)

        test_col3 = [t, f, f, f, t, t]
        col3 = [self.env.is_valid_action(idx) for idx in range(18, 24)]
        self.assertEqual(col3, test_col3)

        test_col4 = [t, t, t, f, t, t]
        col4 = [self.env.is_valid_action(idx) for idx in range(24, 30)]
        self.assertEqual(col4, test_col4)

        test_col5 = [t, t, t, f, t, f]
        col5 = [self.env.is_valid_action(idx) for idx in range(30, 36)]
        self.assertEqual(col5, test_col5)

        test_col6 = [t, t, t, f, t, f]
        col6 = [self.env.is_valid_action(idx) for idx in range(36, 42)]
        self.assertEqual(col6, test_col6)

        test_col7 = [t, t, t, f, t, f]
        col7 = [self.env.is_valid_action(idx) for idx in range(42, 48)]
        self.assertEqual(col7, test_col7)

        test_col8 = [t, t, t, f, t, f]
        col8 = [self.env.is_valid_action(idx) for idx in range(48, 54)]
        self.assertEqual(col8, test_col8)

        test_col9 = [t, t, t, f, t, f]
        col9 = [self.env.is_valid_action(idx) for idx in range(54, 60)]
        self.assertEqual(col9, test_col9)

    def test_action_mask(self):
        t = True
        f = False

        test_col0 = [t, f, f, f, t, f]
        test_col1 = [t, f, f, f, t, f]
        test_col2 = [t, f, f, f, t, f]
        test_col3 = [t, f, f, f, t, t]
        test_col4 = [t, t, t, f, t, t]
        test_col5 = [t, t, t, f, t, f]
        test_col6 = [t, t, t, f, t, f]
        test_col7 = [t, t, t, f, t, f]
        test_col8 = [t, t, t, f, t, f]
        test_col9 = [t, t, t, f, t, f]
        _mask = np.concatenate(
            [
                test_col0,
                test_col1,
                test_col2,
                test_col3,
                test_col4,
                test_col5,
                test_col6,
                test_col7,
                test_col8,
                test_col9
            ]
        )
        test_mask = np.zeros(_mask.shape)
        test_mask[~_mask] = -np.inf

        action_mask = self.env._get_action_mask()
        self.assertTrue((test_mask == action_mask).all())


if __name__ == "__main__":
    unittest.main()
