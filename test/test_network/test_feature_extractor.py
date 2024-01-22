import torch
import unittest
from envs.hex_env import HexEnv
from src.convolution_feature_extractor import ConvolutionFeatureExtractor


def get_model_input_from_observation(obs):
    features = torch.tensor(obs["features"], dtype=torch.float32)
    next_indices = torch.tensor(obs["next"])
    prev_indices = torch.tensor(obs["previous"])
    twin_indices = torch.tensor(obs["twin"])

    return features, next_indices, prev_indices, twin_indices


def batch_observations(features, next_indices, prev_indices, twin_indices, batch_size):
    features = features.repeat(batch_size, 1, 1)
    next_indices = next_indices.repeat(batch_size, 1)
    prev_indices = prev_indices.repeat(batch_size, 1)
    twin_indices = twin_indices.repeat(batch_size, 1)
    return features, next_indices, prev_indices, twin_indices


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.env = HexEnv()
        self.output_features = 16
        self.batch_size = 5
        self.model = ConvolutionFeatureExtractor(
            self.env.observation_space,
            4,
            self.output_features,
            5
        )

    def test_sample_output_features(self):
        obs, info = self.env.reset()
        features, next_indices, prev_indices, twin_indices = get_model_input_from_observation(obs)
        extracted_features = self.model._forward_sample(features, next_indices, prev_indices, twin_indices)
        self.assertEqual(extracted_features.ndim, 2)
        num_samples, num_features = extracted_features.shape
        self.assertEqual(num_samples, 12)
        self.assertEqual(num_features, self.output_features)

    def test_offset_indices(self):
        input_index = torch.tensor([
            [0, 1, 2, 3, -1, -2, 6, 7, 8, 9, -1, -2],
            [0, 5, 2, 4, 9, -1, -2, 3, 8, 7, -1, -2]
        ]
        )
        offset_index = self.model.unroll_and_offset_indices(input_index)
        test_index = torch.tensor(
            [0, 1, 2, 3, -1, -2, 6, 7, 8, 9, -1, -2, 12, 17, 14, 16, 21, -1, -2, 15, 20, 19, -1, -2]
        )
        self.assertTrue((test_index == offset_index).all())

    def test_batch_output_features(self):
        obs, info = self.env.reset()
        features, next_indices, prev_indices, twin_indices = get_model_input_from_observation(obs)
        features, next_indices, prev_indices, twin_indices = batch_observations(
            features,
            next_indices,
            prev_indices,
            twin_indices,
            self.batch_size
        )
        extracted_features = self.model._forward_batch(features, next_indices, prev_indices, twin_indices)
        self.assertEqual(extracted_features.ndim, 3)
        batch_size, sample_size, feature_size = extracted_features.shape
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(sample_size, 12)
        self.assertEqual(feature_size, self.output_features)


if __name__ == "__main__":
    unittest.main()
