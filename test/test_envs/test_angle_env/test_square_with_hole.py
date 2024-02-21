from envs.substep_angle_env import AngleEnv
import unittest
from src.tiler import Tiler


def initialize_graph_and_desired_degree():
    coords = generate_coordinates()
    loop = [
        [0, 1, 2, 3, 0, 4, 7, 6, 5, 4]
    ]
    graph = Tiler.from_face_loops(loop, coords)
    desired_degree = dict(zip(range(8), [2, 2, 2, 2, 4, 4, 4, 4]))

    return graph, desired_degree


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.75, 0.75],
        [0.25, 0.75]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestSquareHoleEnv(unittest.TestCase):
    def setUp(self) -> None:
        env = AngleEnv(4, [9])
        graph, desired_degree = initialize_graph_and_desired_degree()
        env._reset_to_state(graph, desired_degree)
        env.template_center = (0, graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_scores(self):
        self.assertEqual(self.env.template_face_score, 6)
        self.assertEqual(self.env.template_vertex_score, 8)
        self.assertAlmostEqual(self.env.template_angle_score, 8)

    def test_step1(self):
        self.env.step(26)
        self.assertEqual(self.env.template_face_score, 4)
        self.assertEqual(self.env.template_vertex_score, 8)
        self.assertEqual(self.env.template_angle_score, 8)
        self.assertAlmostEqual(self.env.reward, 2 / 3)
        self.assertTrue(self.env.template_center == (0, self.env.graph.half_edge_tag))

    def test_step1(self):
        self.env.step(26)
        self.env.step(44)
        self.env.step(56)

        self.assertEqual(self.env.template_face_score, 0)
        self.assertEqual(self.env.template_vertex_score, 8)
        self.assertEqual(self.env.template_angle_score, 8)
        self.assertAlmostEqual(self.env.reward, 2 / 3)
        self.assertTrue(self.env.template_center == (0, self.env.graph.half_edge_tag))


if __name__ == "__main__":
    unittest.main()
