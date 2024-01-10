from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import sys
import os

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy


def initialize_environment():
    env = RandomPolygonEnv(3, [40])
    return env


env = make_vec_env(
    initialize_environment,
    20
)

feature_extractor_layers = 5
feature_extractor_size = 128
policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(
        input_features=RandomPolygonEnv.get_feature_size(),
        output_features=feature_extractor_size,
        number_of_layers=feature_extractor_layers
    ),
    ortho_init=True
)

ppo_config = dict(
    gamma=0.9987,
    n_steps=256,
    gae_lambda=0.094,
    ent_coef=2.7e-5,
    vf_coef=0.5,
    clip_range=0.178,
    max_grad_norm=0.4117,
    learning_rate=5.5e-5,
    batch_size=32,
)
ppo_config["policy_kwargs"] = policy_kwargs
ppo_config["verbose"] = 1

model = PPO(
    CustomActorCriticPolicy,
    env,
    **ppo_config,
)

total_timesteps = 1_000_000

total_timesteps, callback = model._setup_learn(total_timesteps, None, True, "OPA", False)
callback.on_training_start(locals(), globals())

continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
