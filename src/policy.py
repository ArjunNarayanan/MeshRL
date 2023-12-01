from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from stable_baselines3 import PPO


class CustomNetwork(torch.nn.Module):
    def __init__(self, input_features, output_features=None):
        super().__init__()
        if output_features is None:
            output_features = input_features

        self.latent_dim_pi = output_features
        self.latent_dim_vf = output_features

        # policy network
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_features, output_features),
            torch.nn.LeakyReLU()
        )
        # value network
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(input_features, output_features),
            torch.nn.LeakyReLU()
        )

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
    ):
        self.input_features = kwargs["features_extractor_kwargs"]["output_features"]
        self.output_features = kwargs.get("output_features", self.input_features)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            normalize_images=False,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomNetwork(self.input_features, self.output_features)

    def _get_masked_action_dist_from_latent(self, latent_pi, mask):
        # latent_pi.shape == [batch_size, num_halfedges, num_features]
        assert mask.ndim == 2
        assert latent_pi.ndim == 3
        batch_size, num_halfedges, num_features = latent_pi.shape
        assert mask.shape[0] == batch_size

        action_logits = self.action_net(latent_pi)  # [batch_size, num_halfedges, num_actions_per_halfedge]
        action_logits = action_logits.reshape(batch_size, -1)  # [batch_size, num_actions_per_sample]
        action_logits = action_logits + mask

        return self.action_dist.proba_distribution(action_logits=action_logits)

    @staticmethod
    def _reduce_latent_vf(latent_vf):
        return latent_vf.mean(dim=-2)

    def predict_values(self, obs):
        features = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        latent_vf = self._reduce_latent_vf(latent_vf)
        values = self.value_net(latent_vf)
        return values

    def get_distribution(self, obs):
        features = self.features_extractor(obs)
        mask = obs["mask"]

        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_masked_action_dist_from_latent(latent_pi, mask)
        return distribution

    def evaluate_actions(self, obs, actions):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        mask = obs["mask"]
        distribution = self._get_masked_action_dist_from_latent(latent_pi, mask)
        log_prob = distribution.log_prob(actions)

        latent_vf = self._reduce_latent_vf(latent_vf)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)

        latent_pi, latent_vf = self.mlp_extractor(features)

        latent_vf = self._reduce_latent_vf(latent_vf)
        values = self.value_net(latent_vf)

        mask = obs["mask"]
        distribution = self._get_masked_action_dist_from_latent(latent_pi, mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob
