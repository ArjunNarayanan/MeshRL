import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import LayerNorm
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import Linear


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space,
            input_features,
            output_features,
            sequence_length,
            num_heads,
            number_of_layers,
            dropout
    ):
        super().__init__(observation_space, output_features)
        self.num_input_features = input_features
        self.num_output_features = output_features
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.num_layers = number_of_layers

        self.projector = Linear(input_features, output_features)
        self.positional_encoder = PositionalEncoding(output_features, sequence_length, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=output_features,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        encoder_norm = LayerNorm(output_features)
        self.encoder = TransformerEncoder(
            encoder_layer,
            number_of_layers,
            encoder_norm
        )

    def forward(self, obs):
        x = obs["features"]
        x = self.projector(x)
        x = self.positional_encoder(x)
        x = self.encoder(x)
        return x
