import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy
from src.utils import load_yaml_config


def initialize_environment():
    env_config = config["environment"]
    env = RandomPolygonEnv.from_config(env_config)
    return env


def obs_as_tensor(obs):
    _obs = {}
    for k, v in obs.items():
        v = torch.tensor(v).unsqueeze(0)
        _obs[k] = v
    return _obs


if __name__=="__main__":
    env_config = "../experiments/tiler-random-polygon/triangle/models/tri-5-50-scaled/config.yml"
    config = load_yaml_config(env_config)

    feature_extractor_layers = config["feature_extractor"]["number_of_layers"]
    feature_extractor_size = config["feature_extractor"]["output_features"]
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(
            input_features=RandomPolygonEnv.get_feature_size(),
            output_features=feature_extractor_size,
            number_of_layers=feature_extractor_layers,
        ),
        use_critic=False,
        ortho_init=config["feature_extractor"]["ortho_init"]
    )

    ppo_config = config["PPO"]
    ppo_config["policy_kwargs"] = policy_kwargs
    ppo_config["verbose"] = 1

    env = make_vec_env(
        initialize_environment,
        5,
        vec_env_cls=DummyVecEnv
    )
    model = PPO(
        CustomActorCriticPolicy,
        env,
        **ppo_config,
    )

    model.learn(total_timesteps=1000)
