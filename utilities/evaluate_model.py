import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import sys
import argparse
import math
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.save_util import load_from_zip_file
from src.feature_extractor import feature_extractor_initializer

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.render import Renderer
from src.utils import load_yaml_config


def initialize_eval_environment(polygon_degree):
    env_config = config["environment"]
    env_config["incremental_reward"] = False
    env_config["min_polygon_degree"] = polygon_degree
    env_config["max_polygon_degree"] = polygon_degree
    env = RandomPolygonEnv.from_config(env_config)
    return env


input_folder = "experiments/tiler-random-polygon/triangle/tri-5-50-scaled/"
checkpoint_file = os.path.join(input_folder, "best_model.zip")
config_file = os.path.join(input_folder, "config.yml")

config = load_yaml_config(config_file)

# features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config["feature_extractor"])
# features_extractor_kwargs.update({"input_features": RandomPolygonEnv.get_feature_size()})
# use_critic = config["policy"].get("use_critic", True)
# ortho_init = config["policy"]["ortho_init"]
# policy_kwargs = dict(
#     features_extractor_class=features_extractor_class,
#     features_extractor_kwargs=features_extractor_kwargs,
#     use_critic=use_critic,
#     ortho_init=ortho_init
# )
# ppo_config = config["PPO"]
# ppo_config["policy_kwargs"] = policy_kwargs
# ppo_config = dict(policy_kwargs=policy_kwargs)
# model = PPO.load(checkpoint_file, custom_objects=ppo_config)
model = PPO.load(checkpoint_file)

poly_degrees = list(range(5, 51, 5))
average_rewards = []
std_rewards = []
for poly_degree in poly_degrees:
    env = make_vec_env(lambda: initialize_eval_environment(poly_degree), vec_env_cls=DummyVecEnv, n_envs=10)
    val, std = evaluate_policy(model, env, deterministic=False)
    print("Poly Degree : ", poly_degree, "\tAVG : ", val, "\tSTD : ", std)
    average_rewards.append(val)
    std_rewards.append(std)

