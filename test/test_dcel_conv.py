import unittest

from envs.hex_env import HexEnv
from src.dcel_convolution import DCELConvBlock
# import unittest
import torch


def get_model_input_from_observation(obs):
    features = torch.tensor(obs["features"], dtype=torch.float32)
    next_indices = torch.tensor(obs["next"])
    prev_indices = torch.tensor(obs["previous"])
    twin_indices = torch.tensor(obs["twin"])

    # num_halfedge, num_features = features.shape
    # cycle_boundary_index = num_halfedge
    # twin_boundary_index = cycle_boundary_index + 1
    # next_indices[next_indices < 0] = cycle_boundary_index
    # prev_indices[prev_indices < 0] = cycle_boundary_index
    # twin_indices[twin_indices < 0] = twin_boundary_index

    return features, next_indices, prev_indices, twin_indices


class TestDCELConv(unittest.TestCase):
    def setUp(self):
        self.env = HexEnv()
        self.conv = DCELConvBlock(4, 10)

    def test_conv(self):
        obs, info = self.env.reset()
        features, next_indices, prev_indices, twin_indices = get_model_input_from_observation(obs)
        out = self.conv(features, next_indices, prev_indices, twin_indices)
        num_halfedges, num_features = out.shape
        self.assertEqual((num_halfedges, num_features), (12, 10))




if __name__ == "__main__":
    unittest.main()
