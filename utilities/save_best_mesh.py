import os
import sys
import argparse
import torch
from copy import deepcopy
import glob
import pickle

sys.path.append(os.getcwd())
from envs.environment_maker import initialize_environment
from src.utils import load_yaml_config, load_model_from_checkpoint
from utilities.plot_best_mesh import plot_graph


def generate_environment(polygon_degree):
    env_config = config["environment"]
    init_config = env_config["initializer"]
    init_config["polygon_degree"] = polygon_degree
    env = initialize_environment(env_config)
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
    best_face_score = float("inf")
    best_score = float("inf")

    obs = obs_as_tensor(env._get_obs())
    done = env.is_terminated()

    while not done:
        dist = get_action_distribution(obs)
        obs, done = step_environment(env, dist)
        obs = obs_as_tensor(obs)

        env_face_score = env.get_face_score()
        env_score = env.get_score() / env.get_initial_score()

        if env_face_score < best_face_score:
            best_env = deepcopy(env)
            best_face_score = env_face_score
            best_score = env_score
        elif env_face_score == best_face_score and env_score < best_score:
            best_env = deepcopy(env)
            best_score = env_score

    return best_env, best_face_score, best_score


def get_best_mesh_from_multi_rollout(num_rollouts=10):
    env = generate_environment(polygon_degree)
    initial_env = deepcopy(env)

    best_env = None
    best_face_score = float("inf")
    best_score = float("inf")

    for rollout in range(num_rollouts):
        print("ROLLOUT : ", rollout)
        env.reset()

        rollout_best_env, rollout_face_score, rollout_score = get_best_mesh_from_rollout(env)
        if rollout_face_score < best_face_score:
            best_env = rollout_best_env
            best_score = rollout_score
            best_face_score = rollout_face_score
        elif rollout_face_score == best_face_score and rollout_score < best_score:
            best_env = rollout_best_env
            best_score = rollout_score

    print("\n\tBest Score")
    print("\t\tFace : ", best_face_score)
    print("\t\tTotal: ", best_score)

    return initial_env, best_env


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
    parser.add_argument("-degree", required=True, type=int)
    parser.add_argument("-model", default="best_model.zip")
    parser.add_argument("-config", default="config.yml")
    parser.add_argument("-output", default="best-mesh")
    parser.add_argument("-rollout", default=None)
    parser.add_argument("-trials", default=10, type=int)
    parser.add_argument("-vertex_size", default=20, type=int)
    args = parser.parse_args()

    vertex_size = args.vertex_size
    input_folder = args.input
    checkpoint = args.model
    config_filename = args.config
    polygon_degree = args.degree
    num_trials = args.trials

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

    initial_env, best_env = get_best_mesh_from_multi_rollout(num_rollouts=num_trials)

    print("\tPlotting initial geometry")
    initial_filename = os.path.join(rollout_output_folder, "initial.pdf")
    face_desired_degree = config["environment"]["face_desired_degree"]
    plot_graph(
        initial_env.graph,
        face_desired_degree,
        vertex_desired_degree=initial_env.vertex_desired_degree,
        filename=initial_filename,
        vertex_size=vertex_size
    )

    print("\tPlotting model output")
    best_filename = os.path.join(rollout_output_folder, "coarse.pdf")
    plot_graph(
        best_env.graph,
        face_desired_degree,
        vertex_desired_degree=best_env.vertex_desired_degree,
        filename=best_filename,
        vertex_size=vertex_size
    )

    output_data = dict(
        initial=initial_env,
        best_env=best_env,
    )
    output_file_path = os.path.join(rollout_output_folder, "best_mesh.pkl")
    print("\n\n\tWRITING OUTPUT FILE : ", output_file_path)
    with open(output_file_path, "wb") as output_file:
        pickle.dump(output_data, output_file)
