from envs.angle_env import AngleEnv
import unittest
import numpy as np
from src.tiler import Tiler


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    coords = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, coords)
    return graph


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(6), coords))
    return coords


class TestHexEnvInitial(unittest.TestCase):
    def setUp(self) -> None:
        self.env = AngleEnv(3, [6], template_size=6)
        self.env.graph = initialize_graph()

        self.env.template_center = (3, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env._update_scores_on_reset()

    def test_scores(self):
        self.assertEqual(self.env.average_face_score, 1)
        self.assertEqual(self.env.average_angle_score, 1)

    def test_feature_matrix(self):
        obs = self.env._get_obs()
        coords = obs["features"][:, [0, 1]]
        test_coords = np.array(
            [
                [0, 0],
                [1, 0],
                [-0.5, np.sqrt(3) / 2],
                [1.5, np.sqrt(3) / 2],
                [0, np.sqrt(3)],
                [1, np.sqrt(3)]
            ]
        )
        # self.assertTrue(np.isclose(coords, test_coords, atol=1e-5).all())


class TestHexEnvSteps(unittest.TestCase):
    def setUp(self) -> None:
        self.env = AngleEnv(3, [6], template_size=6)
        self.env.graph = initialize_graph()

        self.env.template_center = (3, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env._update_scores_on_reset()

        self.env._step_insert_edge(5, 2)

    def test_scores(self):
        self.assertAlmostEqual(self.env.average_face_score, 1 / 3)
        self.assertAlmostEqual(self.env.average_angle_score, 0.5)


class TestCompositeActions(unittest.TestCase):
    def setUp(self) -> None:
        self.env = AngleEnv(
            3,
            [6],
            face_reward_weight=0.5,
            angle_reward_weight=0.5,
            vertex_reward_weight=0,
            template_size=6,
            smooth_iterations=0
        )
        self.env.graph = initialize_graph()

        self.env.template_center = (3, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env._update_scores_on_reset()

        self.env._step_insert_edge(0, 2)
        self.env._step_insert_vertex(7)
        self.env._step_insert_edge(8, 1)
        self.env._step_insert_edge(5, 1)

    def test_scores(self):
        self.assertAlmostEqual(self.env.average_face_score, 1 / 6)
        self.assertAlmostEqual(self.env.average_angle_score, 2 / 7)

        reward = self.env.face_reward_weight * (1 / 3 - 1 / 6) + self.env.angle_reward_weight * (1 / 2 - 2 / 7)
        self.assertAlmostEqual(self.env.reward, reward)


if __name__ == "__main__":
    unittest.main()
