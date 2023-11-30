from envs.hex_env_with_insert import HexEnv
import unittest
import numpy as np


class TestActionMask(unittest.TestCase):
    def setUp(self) -> None:
        self.env = HexEnv(
            template_size=8,
            max_actions=1,
            randomize=False,
            incremental_reward=False,
            no_action_reward=0
        )

    def test_valid_action(self):
        test_actions = 6 * [True, True, True, False, True, False] + 12 * [False]
        valid_actions = [self.env.is_valid_action(idx) for idx in range(48)]
        self.assertEqual(test_actions, valid_actions)


class TestSequentialActionMask(unittest.TestCase):
    def setUp(self) -> None:
        env = HexEnv(
            template_size=10,
            max_actions=5,
            randomize=False,
            incremental_reward=False,
            no_action_reward=0
        )
        env._step_insert_edge(1, 1)
        env._step_insert_vertex(6)
        env._step_insert_edge(7, 2)
        env._step_delete_edge(6)
        env._build_template([0])
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

    def test_mask(self):
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


if __name__ == "__main__":
    unittest.main()
