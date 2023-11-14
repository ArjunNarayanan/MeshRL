import torch
from torch.nn import Linear


class DCELConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = Linear(4 * in_channels, out_channels, bias=use_bias)
        self.cycle_boundary_vector = torch.nn.Parameter(torch.randn(self.in_channels))
        self.twin_boundary_vector = torch.nn.Parameter(torch.randn(self.in_channels))

    def forward(self, features, next_indices, prev_indices, twin_indices):
        expanded_features = torch.row_stack([features, self.cycle_boundary_vector, self.twin_boundary_vector])
        next_features = expanded_features[next_indices, :]
        prev_features = expanded_features[prev_indices, :]
        twin_features = expanded_features[twin_indices, :]

        features = torch.cat([features, next_features, prev_features, twin_features], dim=1)
        out_features = self.linear(features)

        return out_features


class DCELConvolution(torch.nn.Module):
    def __init__(self, in_channels, num_convolutions):
        super().__init__()
