from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from envs.hex_env_with_insert import HexEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy


def extract_env(wrapped_env):
    env = wrapped_env.envs[0].env
    return env


def plot_distribution(probs):
    fig, ax = plt.subplots()
    ax.bar(range(len(probs)), probs)
    ax.set_xlabel("Actions")
    ax.set_ylabel("Probabilities")
    ax.grid()
    return fig


def make_output_dir_if_necessary(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


template_size = 18
num_actions_per_halfedge = 6
env = HexEnv(template_size=template_size)

feature_extractor_size = 128
feature_extractor_layers = 2
num_envs = 6
env = make_vec_env(HexEnv, num_envs)


num_actions = template_size * num_actions_per_halfedge
output_dir = "/home/anarayan/Research/MeshRL/Polygraph/experiments/hex_env_with_insert"
make_output_dir_if_necessary(output_dir)

policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(
        input_features=5,
        output_features=feature_extractor_size,
        number_of_layers=feature_extractor_layers
    )
)

model = PPO(
    CustomActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    ent_coef=0.01,
    tensorboard_log=output_dir
)

eval_env = HexEnv(template_size=template_size)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=output_dir,
    eval_freq=1000,
    deterministic=True,
    render=False
)

if __name__ == "__main__":
    model.learn(total_timesteps=1000000, callback=eval_callback)
