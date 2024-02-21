from envs.substep_angle_env import AngleEnv
import unittest
from src.tiler import Tiler


def initialize_graph_and_desired_degree():
    coords = generate_coordinates()
    loop = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 4]
    ]
    graph = Tiler.from_face_loops(loop, coords)
    desired_degree = dict(zip(range(9), [2, 2, 2, 2, 3, 5, 3, 5, 3]))

    return graph, desired_degree


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0., 0.5],
        [0.25, 0.5],
        [0.5, 0.5 + 1e-9],
        [0.75, 0.5],
        [0.5, 0.5 - 1e-9]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestCenterCrack(unittest.TestCase):
    def setUp(self) -> None:
        env = AngleEnv(4, [9], template_size=30, smooth_iterations=40)
        graph, desired_degree = initialize_graph_and_desired_degree()
        env._reset_to_state(graph, desired_degree)
        env.template_center = (0, graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_scores(self):
        self.assertEqual(self.env.template_face_score, 7)
        self.assertEqual(self.env.template_vertex_score, 7)
        self.assertAlmostEqual(self.env.template_angle_score, 7)

    def test_step(self):
        self.env._step_half_edge_action(0, 4)
        self.env._step_half_edge_action(0, 4)
        self.env._step_half_edge_action(0, 4)

        self.env._step_half_edge_action(2, 4)
        self.env._step_half_edge_action(2, 4)
        self.env._step_half_edge_action(2, 4)

        self.env._step_half_edge_action(1, 4)

        self.env._step_half_edge_action(9, 2)
        self.env._step_half_edge_action(8, 2)
        self.env._step_half_edge_action(7, 2)
        self.env._step_half_edge_action(23, 2)
        self.env._step_half_edge_action(25, 2)
        self.env._step_half_edge_action(6, 2)
        self.env._step_half_edge_action(5, 2)

        self.env._local_reset_template_center()
        self.env._build_template()
        self.env._update_scores_on_step()

        self.assertEqual(self.env.template_face_score, 0)
        self.assertEqual(self.env.template_vertex_score, 0)
        self.assertAlmostEqual(self.env.template_angle_score, 0, places=4)


if __name__ == "__main__":
    unittest.main()
