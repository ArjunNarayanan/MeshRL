import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import sys
import argparse
import math

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.render import Renderer
from src.utils import load_yaml_config


def plot_distribution(probs):
    fig, ax = plt.subplots()
    xtick_locs = range(len(probs))

    ax.bar(xtick_locs, probs)
    # ax.set_xticks(xtick_locs)
    ax.set_xlabel("Actions")
    ax.set_ylabel("Probabilities")
    ax.grid()
    return fig


def initialize_environment():
    env_config = config["environment"]
    env = RandomPolygonEnv.from_config(env_config)
    return env


def get_action_distribution(obs, plot=False):
    # template_halfedges = env.index_to_halfedge
    # print("Template : ", template_halfedges)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
    probs = dist.distribution.probs[0]
    if plot:
        plot_distribution(probs)
    return dist


def step_environment(env, dist):
    action = dist.get_actions()
    action = action.cpu().numpy()
    linear_action = action[0]
    action_probability = dist.distribution.probs[0, linear_action].item()

    action_halfedge, action_type = env._linear_action_index_to_halfedge_and_action(action[0])

    print("Step : ", env.num_steps)
    # print("Selected action : ", action[0])
    # print("Selected action probability : ", action_probability)
    # print("\tHalfedge: ", action_halfedge, " \tType: ", action_type)
    obs, reward, done, truncated, info = env.step(action[0])
    print("Reward : ", reward)
    # print("Terminated : ", done)

    return obs, done


def obs_as_tensor(obs):
    _obs = {}
    for k, v in obs.items():
        v = torch.tensor(v).unsqueeze(0)
        _obs[k] = v
    return _obs


def make_output_folder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def reset_if_done(env, obs, done):
    if done:
        total_reward = abs(env.initial_score - env.score)/env.initial_score
        print("\nNORMALIZED RETURN : ", total_reward, "\n")
        env.reset()
        # renderer.graph = env.graph
        # renderer.coords = env.graph.vertex_coordinates
        obs = obs_as_tensor(env._get_obs())
        return obs
    else:
        return obs_as_tensor(obs)


def plot_env(env, filename=None):
    renderer = Renderer(env.graph, env.graph.vertex_coordinates)
    renderer.coords = env.graph.vertex_coordinates
    renderer.plot()
    renderer.plot_vertex_scores(env.vertex_desired_degree)
    renderer.plot_face_scores(env.face_desired_degree)
    if filename is not None:
        renderer.fig.savefig(filename)


def plot_rollout(rollout):
    env = initialize_environment()
    fig_output_folder = os.path.join(input_folder, "figures", "rollout-" + str(rollout))
    make_output_folder(fig_output_folder)

    step = 0
    figname = "step-" + str(step).zfill(NUM_DIGITS) + ".png"
    output_file = os.path.join(fig_output_folder, figname)
    plot_env(env, filename=output_file)
    obs = obs_as_tensor(env._get_obs())
    done = env.is_terminated()

    while not done:
        dist = get_action_distribution(obs)
        obs, done = step_environment(env, dist)
        obs = obs_as_tensor(obs)
        env.graph.laplace_smooth_vertices()

        figname = "step-" + str(step).zfill(NUM_DIGITS) + ".png"
        output_file = os.path.join(fig_output_folder, figname)

        plot_env(env, filename=output_file)
        print("Saving figure : ", output_file)
        step += 1

    total_reward = abs(env.initial_score - env.score) / env.initial_score
    print("\nNORMALIZED RETURN : ", total_reward, "\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True)
    parser.add_argument("-rollout", required=True)
    parser.add_argument("-degree", default=None, type=int)

    args = parser.parse_args()
    input_folder = args.input
    rollout = args.rollout
    checkpoint = os.path.join(input_folder, "best_model.zip")
    model = PPO.load(checkpoint)

    polygon_degree = args.degree
    config_fn = os.path.join(input_folder, "config.yml")
    config = load_yaml_config(config_fn)
    if polygon_degree is not None:
        config["environment"]["min_polygon_degree"] = polygon_degree
        config["environment"]["max_polygon_degree"] = polygon_degree

    env_config = config["environment"]
    max_steps = env_config["max_steps_factor"] * env_config["max_polygon_degree"]
    NUM_DIGITS = int(math.log10(max_steps))

    plot_rollout(rollout)
