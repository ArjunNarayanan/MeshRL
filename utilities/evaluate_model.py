import os
import sys
import argparse
import math
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.utils import load_yaml_config, load_model_from_checkpoint
from src.plot import plot_returns_vs_poly_degree


def initialize_eval_environment(polygon_degree, max_steps_factor=None):
    env_config = config["environment"]
    env_config["incremental_reward"] = False
    env_config["min_polygon_degree"] = polygon_degree
    env_config["max_polygon_degree"] = polygon_degree
    env_config["eval_mode"] = True
    env_config["fixed_reset"] = True
    if max_steps_factor is not None:
        env_config["max_steps_factor"] = max_steps_factor
    env = RandomPolygonEnv.from_config(env_config)
    return env


def get_best_returns_for_size(model, poly_degree, num_trials=10, n_eval_episodes=10, max_steps_factor=None):
    best_returns = []
    for trial in range(num_trials):
        env = make_vec_env(
            lambda: initialize_eval_environment(poly_degree, max_steps_factor),
            vec_env_cls=DummyVecEnv,
            n_envs=10,
        )
        ep_vals, ep_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
            return_episode_rewards=True
        )
        best_returns.append(max(ep_vals))
    return best_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="input folder", required=True)
    parser.add_argument("-min", help="min polygon degree", type=int, required=True)
    parser.add_argument("-max", help="max polygon degree", type=int, required=True)
    parser.add_argument("-step", help="step size for polygon degree", type=int, default=5)
    parser.add_argument("-model", help="model file name", default="best_model.zip")
    parser.add_argument("-config", help="config file name", default="config.yml")

    args = parser.parse_args()

    input_folder = args.input
    checkpoint_file = os.path.join(input_folder, args.model)
    config_file = os.path.join(input_folder, args.config)
    config = load_yaml_config(config_file)

    model = load_model_from_checkpoint(checkpoint_file, config_file)

    avg_returns, std_returns = [], []
    poly_degrees = list(range(args.min, args.max + 1, args.step))
    for poly_degree in poly_degrees:
        best_returns = get_best_returns_for_size(model, poly_degree, max_steps_factor=3)
        avg = np.mean(best_returns)
        std = np.std(best_returns)
        print("Poly : ", poly_degree, "\tAVG : ", avg, "\tSTD : ", std)
        avg_returns.append(avg)
        std_returns.append(std)

    fig = plot_returns_vs_poly_degree(poly_degrees, avg_returns, std_returns)
    outputfolder = os.path.join(input_folder, "output")
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)

    outputfile = os.path.join(outputfolder, "avg_returns.png")
    fig.savefig(outputfile)
