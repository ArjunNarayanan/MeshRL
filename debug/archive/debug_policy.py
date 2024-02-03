import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.feature_extractor import feature_extractor_initializer
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


env_config = "experiments/tiler-random-polygon/triangle/models/tri-5-50-convolution/config.yml"
config = load_yaml_config(env_config)

features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config["feature_extractor"])
features_extractor_kwargs.update({"input_features": RandomPolygonEnv.get_feature_size()})
policy_kwargs = dict(
    features_extractor_class=features_extractor_class,
    features_extractor_kwargs=features_extractor_kwargs,
    use_critic=config["policy"]["use_critic"],
    ortho_init=config["policy"]["ortho_init"]
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

# model.learn(total_timesteps=1000)
