from envs.global_angle_env import AngleEnv
import unittest
from envs.environment_initializers import SquareHole


class TestSquareHoleEnv(unittest.TestCase):
    def setUp(self) -> None:
        initializer = SquareHole(90)
        env = AngleEnv(4, initializer)
        env.template_center = (0, env.graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_scores(self):
        self.assertEqual(self.env.global_face_score, 6)
        self.assertEqual(self.env.global_vertex_score, 8)
        self.assertAlmostEqual(self.env.global_angle_score, 8)

    def test_step1(self):
        self.env.step(26)
        self.assertEqual(self.env.global_face_score, 4)
        self.assertEqual(self.env.global_vertex_score, 8)
        self.assertEqual(self.env.global_angle_score, 8)
        self.assertAlmostEqual(self.env.reward, 1 / 9)

    def test_step2(self):
        self.env.step(26)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env.step(44)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env.step(56)

        self.assertEqual(self.env.global_face_score, 0)
        self.assertEqual(self.env.global_vertex_score, 8)
        self.assertEqual(self.env.global_angle_score, 8)
        self.assertAlmostEqual(self.env.reward, 1 / 9)


if __name__ == "__main__":
    unittest.main()
