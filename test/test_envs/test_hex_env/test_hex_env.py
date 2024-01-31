from envs.hex_env import HexEnv
import unittest
import numpy as np


class TestInsertHalfedge(unittest.TestCase):
    def setUp(self) -> None:
        self.env = HexEnv()

    def test_feature_matrix(self):
        matrix = self.env._get_feature_matrix()
        test_matrix = np.zeros((12, 4))
        test_matrix[0:6, :] = [2 / 3, 3, 2, 3]
        self.assertTrue(np.all(matrix == test_matrix))

    def test_next_edges(self):
        next_edges = self.env._get_next_edges()
        test_edges = [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11]
        self.assertTrue(np.all(next_edges == test_edges))

    def test_previous_edges(self):
        prev_edges = self.env._get_previous_edges()
        test_edges = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
        self.assertTrue(np.all(prev_edges == test_edges))

    def test_twin_edges(self):
        twin_edges = self.env._get_twin_edges()
        self.assertTrue(len(twin_edges) == 12)
        self.assertTrue(np.all(twin_edges == -1))

    def test_reset(self):
        obs, info = self.env.reset()

        self.assertEqual(self.env.num_actions, 0)

        feature_matrix = np.zeros((12, 4))
        feature_matrix[0:6, :] = [2 / 3, 3, 2, 3]
        self.assertTrue(np.all(obs["features"] == feature_matrix))

        next_edges = [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11]
        self.assertTrue(np.all(obs["next"] == next_edges))

        prev_edges = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
        self.assertTrue(np.all(obs["previous"] == prev_edges))

        self.assertTrue(len(obs["twin"] == 12))
        self.assertTrue(np.all(obs["twin"] == -1))

        self.assertTrue(info["score"] == 3)

    def test_step0(self):
        obs, reward, terminated, truncated, info = self.env.step(0)

        self.assertEqual(self.env.num_actions, 1)
        self.assertEqual(self.env.score, 2)

        self.assertEqual(reward, 1)
        self.assertEqual(terminated, False)
        self.assertEqual(self.env.graph.number_of_halfedges(), 8)
        self.assertEqual(self.env.graph.number_of_faces(), 2)
        self.assertEqual(self.env.graph.number_of_vertices(), 6)

        htag = self.env.graph.halfedge_tag
        test_next = [1, 6, 3, 4, 5, 7, 0, 2]
        self.assertTrue(
            all(self.env.graph.next_halfedge(hidx) == (ne, htag) for (hidx, ne) in zip(range(8), test_next))
        )
        test_prev = [6, 0, 7, 2, 3, 4, 1, 5]
        self.assertTrue(
            all(self.env.graph.previous_halfedge(hidx) == (pe, htag) for (hidx, pe) in zip(range(8), test_prev))
        )

        vtag = self.env.graph.vertex_tag
        test_src = [0, 1, 2, 3, 4, 5, 2, 0]
        self.assertTrue(all(self.env.graph.source_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(8), test_src)))
        test_target = [1, 2, 3, 4, 5, 0, 0, 2]
        self.assertTrue(
            all(self.env.graph.target_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(8), test_target))
        )

        btag = self.env.graph.boundary_tag
        twin_idx = [0, 1, 2, 3, 4, 5, 7, 6]
        twin_tags = 6 * [btag] + 2 * [htag]
        self.assertTrue(all(
            self.env.graph.twin_halfedge(hidx) == (t, tag) for (hidx, t, tag) in zip(range(8), twin_idx, twin_tags))
        )

    def test_step8(self):
        obs, reward, terminated, truncated, info = self.env.step(7)

        self.assertEqual(self.env.num_actions, 1)
        self.assertEqual(self.env.score, 2)

        self.assertEqual(self.env.graph.number_of_halfedges(), 8)
        self.assertEqual(self.env.graph.number_of_faces(), 2)
        self.assertEqual(self.env.graph.number_of_vertices(), 6)
        self.assertEqual(reward, 1)
        self.assertEqual(terminated, False)

        test_features = np.zeros((12, 4))
        test_features[:8, 0] = [2 / 3, 2 / 3, 1, 2 / 3, 2 / 3, 1, 1, 1]
        test_features[:8, 1] = 3
        test_features[:8, 2] = 4 / 3
        test_features[:8, 3] = 3
        self.assertTrue(np.all(test_features == obs["features"]))

        test_next = np.arange(12)
        test_next[:8] = [1, 7, 3, 4, 6, 0, 2, 5]
        self.assertTrue(np.all(test_next == obs["next"]))

        test_prev = np.arange(12)
        test_prev[:8] = [5, 0, 6, 2, 3, 7, 4, 1]
        self.assertTrue(np.all(test_prev == obs["previous"]))

        test_twin = np.repeat(-1, 12)
        test_twin[[6, 7]] = [7, 6]
        self.assertTrue(np.all(test_twin == obs["twin"]))

    def test_step17_step5(self):
        obs1, reward1, terminated1, truncated, info = self.env.step(17)
        obs2, reward2, terminated2, truncated, info = self.env.step(4)
        obs3, reward3, terminated3, truncated, info = self.env.step(33)

        self.assertEqual(self.env.num_actions, 3)
        self.assertEqual(self.env.score, 1)

        self.assertEqual(self.env.graph.number_of_halfedges(), 10)
        self.assertEqual(self.env.graph.number_of_faces(), 3)
        self.assertEqual(self.env.graph.number_of_vertices(), 6)

        self.assertEqual(reward1, 1)
        self.assertEqual(reward2, 1)
        self.assertEqual(reward3, -1)

        self.assertFalse(terminated1)
        self.assertFalse(terminated2)

        test_features = np.zeros((12, 4))
        test_features[:10, 0] = [2 / 3, 1, 2 / 3, 1, 2 / 3, 4 / 3, 1, 4 / 3, 4 / 3, 1]
        test_features[:10, 1] = 3
        test_features[:10, 2] = [1, 4 / 3, 4 / 3, 1, 1, 1, 4 / 3, 1, 4 / 3, 1]
        test_features[:10, 3] = 3
        self.assertTrue(np.all(test_features == obs2["features"]))

        test_next = np.arange(12)
        test_next[:10] = [9, 2, 6, 4, 7, 0, 8, 3, 1, 5]
        self.assertTrue(np.all(test_next == obs2["next"]))

        test_prev = np.arange(12)
        test_prev[:10] = [5, 8, 1, 7, 3, 9, 2, 4, 6, 0]
        self.assertTrue(np.all(test_prev == obs2["previous"]))

        test_twin = np.repeat(-1, 12)
        test_twin[[6, 7, 8, 9]] = [7, 6, 9, 8]
        self.assertTrue(np.all(test_twin == obs2["twin"]))

    def test_optimum_score(self):
        obs1, reward1, terminated1, truncated, info = self.env.step(17)
        obs2, reward2, terminated2, truncated, info = self.env.step(4)
        obs3, reward3, terminated3, truncated, info = self.env.step(18)

        self.assertEqual(self.env.num_actions, 3)
        self.assertEqual(self.env.score, 0)
        self.assertEqual(reward3, 1)
        self.assertTrue(terminated3)

    def test_invalid_action(self):
        for step in range(self.env.max_actions - 1):
            obs, reward, terminated, truncated, info = self.env.step(18)
            self.assertEqual(self.env.num_actions, step + 1)
            self.assertEqual(self.env.score, 3)
            self.assertEqual(reward, -1)
            self.assertFalse(terminated)

        obs, reward, terminated, truncated, info = self.env.step(25)
        self.assertEqual(self.env.num_actions, self.env.max_actions)
        self.assertEqual(self.env.score, 3)
        self.assertEqual(reward, -1)
        self.assertTrue(terminated)


if __name__ == "__main__":
    unittest.main()
