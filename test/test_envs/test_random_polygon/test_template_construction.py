from envs.random_polygon_tiler_env import RandomPolygonEnv
import unittest
import numpy as np
from envs.environment_initializers import MixedMesh
from src.render import Renderer


class TestTemplate(unittest.TestCase):
    def setUp(self) -> None:
        init = MixedMesh(90)
        self.env = RandomPolygonEnv(4, init, template_size=10)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()

    def test_next_edges(self):
        next_indices = self.env._get_next_edges()
        b = self.env.template_boundary_index
        test_next_indices = [1, 4, 0, 6, 8, 2, b, 3, 5, b]
        self.assertTrue((next_indices == test_next_indices).all())
    
    def test_previous_edges(self):
        prev_indices = self.env._get_previous_edges()
        b = self.env.template_boundary_index
        test_prev_indices = [2, 0, 5, 7, 1, 8, 3, b, 4, b]
        self.assertTrue((prev_indices == test_prev_indices).all())

    def test_twin_edges(self):
        twin_indices = self.env._get_twin_edges()
        t = self.env.template_boundary_index
        g = self.env.geometric_boundary_index
        test_twin_indices = [3, g, g, 0, 9, g, g, g, g, 4]
        self.assertTrue((twin_indices == test_twin_indices).all())


if __name__ == "__main__":
    unittest.main()
