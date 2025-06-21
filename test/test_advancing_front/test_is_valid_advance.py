import unittest
import numpy as np
from src.tiler import Tiler
from src.advancing_front import AdvancingFront

class TestIsValidAdvance(unittest.TestCase):
    def setUp(self):
        # Create a simple triangle mesh
        face_loops = [[0, 1, 2]]
        coords = {
            0: np.array([0.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
        }
        self.tiler = Tiler.from_face_loops(face_loops, vertex_coordinates=coords)
        self.front = AdvancingFront(self.tiler)
        # The half-edge from 0 to 1 is (0, 'h')
        self.hidx = (0, self.tiler.half_edge_tag)

    def test_valid_advance(self):
        # Point above edge 0->1 (should be valid)
        new_coord = np.array([0.5, 0.2])
        self.assertTrue(self.front.is_valid_advance(self.hidx, new_coord))

    def test_invalid_advance_right(self):
        # Point below edge 0->1 (should be invalid)
        new_coord = np.array([0.5, -1.0])
        self.assertFalse(self.front.is_valid_advance(self.hidx, new_coord))

    def test_colinear(self):
        # Point colinear with 0->1 (should be invalid)
        new_coord = np.array([2.0, 0.0])
        self.assertFalse(self.front.is_valid_advance(self.hidx, new_coord))

if __name__ == "__main__":
    unittest.main() 