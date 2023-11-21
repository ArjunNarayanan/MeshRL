from envs.hex_env_with_insert import HexEnv
import unittest


class TestHexEnv(unittest.TestCase):
    def setUp(self):
        self.env = HexEnv(False)

    def test_insert_edge_reward(self):
        self.env._step_insert_edge(0, 2)
        self.assertEqual(self.env.reward, 3)

    def test_vertex_reward(self):
        self.env._step_insert_vertex(4)
        self.assertEqual(self.env.reward, -3)

    def test_no_action_reward(self):
        self.env._step_insert_edge(0, 8)
        self.assertEqual(self.env.reward, self.env.no_action_reward)

    def test_delete_edge_score(self):
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)

        self.env._step_insert_edge(2, 2)
        self.assertEqual(self.env.reward, 3)
        self.assertEqual(self.env.global_score(), 6)
        self.assertEqual(self.env.score, 6)

        self.env._step_insert_vertex(6)
        self.assertEqual(self.env.reward, -6)
        self.assertEqual(self.env.global_score(), 12)
        self.assertEqual(self.env.score, 12)

        self.env._step_insert_edge(8, 2)
        self.assertEqual(self.env.reward, 3)
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)

        self.env._step_delete_edge(8)
        self.assertEqual(self.env.reward, -3)
        self.assertEqual(self.env.global_score(), 12)
        self.assertEqual(self.env.score, 12)

        self.env._step_delete_edge(10)
        self.assertEqual(self.env.reward, self.env.no_action_reward)
        self.assertEqual(self.env.global_score(), 12)
        self.assertEqual(self.env.score, 12)

        self.env._step_delete_edge(6)
        self.assertEqual(self.env.reward, self.env.no_action_reward)
        self.assertEqual(self.env.global_score(), 12)
        self.assertEqual(self.env.score, 12)

    def test_vertex_delete_reward(self):
        self.env._step_insert_vertex(5)
        self.assertEqual(self.env.global_score(), 12)
        self.assertEqual(self.env.score, 12)
        self.assertEqual(self.env.reward, -3)

        self.env._step_insert_edge(5, 3)
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)
        self.assertEqual(self.env.reward, 3)

        self.env._step_delete_source_vertex(6)
        self.assertEqual(self.env.global_score(), 6)
        self.assertEqual(self.env.score, 6)
        self.assertEqual(self.env.reward, 3)

        self.env._step_delete_source_vertex(7)
        self.assertEqual(self.env.global_score(), 6)
        self.assertEqual(self.env.score, 6)
        self.assertEqual(self.env.reward, self.env.no_action_reward)

        self.env._step_insert_vertex(7)
        self.assertEqual(self.env.global_score(), 12)
        self.assertEqual(self.env.score, 12)
        self.assertEqual(self.env.reward, -6)

        self.env._step_insert_edge(8, 2)
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)
        self.assertEqual(self.env.reward, 3)

        self.env._step_delete_source_vertex(9)
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)
        self.assertEqual(self.env.reward, self.env.no_action_reward)

        self.env._step_delete_source_vertex(8)
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)
        self.assertEqual(self.env.reward, self.env.no_action_reward)

    def test_composite_reward(self):
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)

        self.env._step_insert_edge(5, 1)
        self.assertEqual(self.env.reward, 3)
        self.assertEqual(self.env.global_score(), 6)
        self.assertEqual(self.env.score, 6)

        self.env._step_insert_vertex(7)
        self.assertEqual(self.env.reward, -6)
        self.assertEqual(self.env.global_score(), 12)
        self.assertEqual(self.env.score, 12)

        self.env._step_insert_edge(6, 1)
        self.assertEqual(self.env.reward, 3)
        self.assertEqual(self.env.global_score(), 9)
        self.assertEqual(self.env.score, 9)

        self.env._step_insert_edge(2, 3)
        self.assertEqual(self.env.reward, 3)
        self.assertEqual(self.env.global_score(), 6)
        self.assertEqual(self.env.score, 6)

        self.env._step_insert_edge(12, 2)
        self.assertEqual(self.env.reward, 3)
        self.assertEqual(self.env.global_score(), 3)
        self.assertEqual(self.env.score, 3)

        self.env._step_insert_edge(3, 1)
        self.assertEqual(self.env.reward, 3)
        self.assertEqual(self.env.global_score(), 0)
        self.assertEqual(self.env.score, 0)
