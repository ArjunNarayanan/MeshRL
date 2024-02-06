import os
import sys
import argparse
import torch
from copy import deepcopy
import glob
import pickle

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.utils import load_yaml_config, load_model_from_checkpoint


def initialize_environment(polygon_degree, max_steps_factor=None):
    env_config = config["environment"]
    env_config["min_polygon_degree"] = polygon_degree
    env_config["max_polygon_degree"] = polygon_degree
    env_config["fixed_reset"] = True
    if max_steps_factor is not None:
        env_config["max_steps_factor"] = max_steps_factor
    env = RandomPolygonEnv.from_config(env_config)
    return env


def obs_as_tensor(obs):
    _obs = {}
    for k, v in obs.items():
        v = torch.tensor(v).unsqueeze(0)
        _obs[k] = v
    return _obs


def get_action_distribution(obs):
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)

    return dist


def step_environment(env, dist):
    action = dist.get_actions()
    action = action.cpu().numpy()

    obs, reward, done, truncated, info = env.step(action[0])

    return obs, done


def get_best_mesh_from_rollout(env):
    best_env = None
    best_score = float("inf")

    obs = obs_as_tensor(env._get_obs())
    done = env.is_terminated()

    while not done:
        dist = get_action_distribution(obs)
        obs, done = step_environment(env, dist)
        obs = obs_as_tensor(obs)

        if env.face_score == 0 and env.vertex_score < best_score:
            best_env = deepcopy(env)
            best_score = env.vertex_score

    return best_env, best_score


def get_best_mesh_from_multi_rollout(num_rollouts=10):
    env = initialize_environment(polygon_degree, max_steps_factor)

    best_env = None
    best_score = float("inf")
    initial_env = deepcopy(env)

    for rollout in range(num_rollouts):
        print("ROLLOUT : ", rollout)
        env.reset()

        rollout_best_env, score = get_best_mesh_from_rollout(env)
        if score < best_score:
            print("\tNew Best Score!")
            best_env = rollout_best_env
            best_score = score

        print("\tScore : ", score)

    return initial_env, best_env, best_score


def get_next_rollout_index():
    rollout_dirs = glob.glob(os.path.join(output_folder, "rollout-*"))
    rollout_names = [os.path.basename(f) for f in rollout_dirs]
    rollout_indices = [int(name.split("-")[1]) for name in rollout_names]
    if rollout_indices:
        return max(rollout_indices) + 1
    else:
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True)
    parser.add_argument("-degree", default=None, type=int)
    parser.add_argument("-model", default="best_model.zip")
    parser.add_argument("-config", default="config.yml")
    parser.add_argument("-output", default="best-mesh")
    parser.add_argument("-steps", help="max step factor", default=3, type=int)
    parser.add_argument("-rollout", default=None)
    args = parser.parse_args()

    input_folder = args.input
    checkpoint = args.model
    config_filename = args.config
    polygon_degree = args.degree
    max_steps_factor = args.steps

    print("Generating best mesh for polygon degree : ", polygon_degree)

    output_folder = os.path.join(input_folder, args.output)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if not args.rollout:
        rollout_index = get_next_rollout_index()
    else:
        rollout_index = args.rollout

    rollout_output_folder = os.path.join(output_folder, "rollout-" + str(rollout_index))
    if not os.path.isdir(rollout_output_folder):
        os.makedirs(rollout_output_folder)

    checkpoint_file = os.path.join(input_folder, checkpoint)
    config_file = os.path.join(input_folder, config_filename)

    model = load_model_from_checkpoint(checkpoint_file, config_file)
    config = load_yaml_config(config_file)

    initial_env, best_env, best_score = get_best_mesh_from_multi_rollout()

    output_data = dict(
        initial=initial_env,
        best_env=best_env,
        best_score=best_score
    )
    output_file_path = os.path.join(rollout_output_folder, "best_mesh.pkl")
    print("\n\n\tWRITING OUTPUT FILE : ", output_file_path)
    print("\tBEST SCORE : ", best_score)
    with open(output_file_path, "wb") as output_file:
        pickle.dump(output_data, output_file)
