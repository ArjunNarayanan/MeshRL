import os
import sys
import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

sys.path.append(os.getcwd())
from envs.environment_maker import initialize_environment
from src.utils import load_yaml_config, load_model_from_checkpoint
import pandas as pd


def initialize_eval_environment(polygon_degree):
    env_config = config["environment"]
    init_config = env_config["initializer"]
    init_config["polygon_degree"] = polygon_degree
    env = initialize_environment(env_config)
    return env


def get_best_returns_for_size(model, poly_degree, num_trials=10, n_eval_episodes=10):
    best_returns = []
    for trial in range(num_trials):
        env = make_vec_env(
            lambda: initialize_eval_environment(poly_degree),
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
    parser.add_argument("-step", help="step size for polygon degree", type=int, default=2)
    parser.add_argument("-model", help="model file name", default="best_model.zip")
    parser.add_argument("-config", help="config file name", default="config.yml")
    parser.add_argument("-output", help="output folder name", default="evaluation")

    args = parser.parse_args()

    input_folder = args.input
    checkpoint_file = os.path.join(input_folder, args.model)
    config_file = os.path.join(input_folder, args.config)
    min_poly = args.min
    max_poly = args.max
    step_poly = args.step
    outputfolder = os.path.join(input_folder, args.output)

    config = load_yaml_config(config_file)
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)

    model = load_model_from_checkpoint(checkpoint_file, config_file)

    avg_returns, std_returns = [], []
    poly_degrees = list(range(min_poly, max_poly, step_poly))
    for poly_degree in poly_degrees:
        best_returns = get_best_returns_for_size(model, poly_degree)
        avg = np.mean(best_returns)
        std = np.std(best_returns)
        print("Poly : ", poly_degree, "\tAVG : ", avg, "\tSTD : ", std)
        avg_returns.append(avg)
        std_returns.append(std)

    data = {
        "polygon degree": poly_degrees,
        "average returns": avg_returns,
        "std. deviation returns": std_returns
    }
    df = pd.DataFrame(data)

    out_file_name = "eval-" + str(min_poly) + "-" + str(max_poly) + ".csv"
    output_file = os.path.join(outputfolder, out_file_name)
    print("\n\nWriting CSV data log")
    df.to_csv(output_file, index=False)
