import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import Linear
from src.dcel_convolution import RecurrentConvolution


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, input_features, output_features, number_of_layers):
        super().__init__(observation_space, output_features)
        self.input_features = input_features
        self.output_features = output_features
        self.number_of_layers = number_of_layers

        self.projector = Linear(input_features, output_features)
        self.convolution = RecurrentConvolution(output_features, number_of_layers)

    def _forward_sample(self, features, next_indices, prev_indices, twin_indices):
        # check shape consistency
        assert features.ndim == 2
        assert next_indices.ndim == prev_indices.ndim == twin_indices.ndim == 1
        sample_size, feature_size = features.shape
        assert feature_size == self.input_features
        assert next_indices.shape[0] == prev_indices.shape[0] == twin_indices.shape[0] == sample_size

        features = self.projector(features)
        features = self.convolution(features, next_indices, prev_indices, twin_indices)

        return features

    @staticmethod
    def unroll_and_offset_indices(batched_indices: torch.Tensor):
        assert batched_indices.ndim == 2
        num_batch, skip = batched_indices.shape
        offset = skip
        offset_indices = [batched_indices[0].clone()]
        for batch in range(1, num_batch):
            indices = batched_indices[batch].clone()
            indices[indices >= 0] += offset
            offset_indices.append(indices)
            offset += skip

        offset_indices = torch.cat(offset_indices)
        return offset_indices

    def _forward_batch(self, features, next_indices, prev_indices, twin_indices):
        # Check input shape consistency
        assert features.ndim == 3
        assert next_indices.ndim == prev_indices.ndim == twin_indices.ndim == 2
        batch_size, template_size, feature_size = features.shape
        assert feature_size == self.input_features
        assert next_indices.shape == torch.Size([batch_size, template_size])
        assert prev_indices.shape == torch.Size([batch_size, template_size])
        assert twin_indices.shape == torch.Size([batch_size, template_size])

        unrolled_features = features.reshape(-1, feature_size)
        offset_next = self.unroll_and_offset_indices(next_indices)
        offset_prev = self.unroll_and_offset_indices(prev_indices)
        offset_twin = self.unroll_and_offset_indices(twin_indices)

        extracted_features = self._forward_sample(unrolled_features, offset_next, offset_prev, offset_twin)
        extracted_features = extracted_features.reshape(batch_size, -1, self.output_features)

        return extracted_features

    def forward(self, obs):
        features = obs["features"]
        next_indices = obs["next"].round().long()
        prev_indices = obs["previous"].round().long()
        twin_indices = obs["twin"].round().long()
        if features.ndim == 3:
            return self._forward_batch(features, next_indices, prev_indices, twin_indices)
        elif features.ndim == 2:
            return self._forward_sample(features, next_indices, prev_indices, twin_indices)
        else:
            raise ValueError("Expected features.ndim == 2 or 3, got ", features.ndim)
