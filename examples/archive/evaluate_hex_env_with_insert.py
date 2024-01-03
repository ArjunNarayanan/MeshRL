import torch
from stable_baselines3 import PPO
from envs.hex_env_with_insert import HexEnv
from src.render import Renderer
import matplotlib.pyplot as plt
import os
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
    env = HexEnv.from_config(env_config)
    return env


def get_action_distribution(obs):
    template_halfedges = env.index_to_halfedge
    print("Template : ", template_halfedges)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
    probs = dist.distribution.probs[0]
    plot_distribution(probs)
    return dist


def step_environment(dist):
    action = dist.get_actions()
    action = action.cpu().numpy()

    action_halfedge, action_type = env._linear_action_index_to_halfedge_and_action(action[0])

    print("Selected action : ", action[0])
    print("\tHalfedge: ", action_halfedge, " \tType: ", action_type)
    obs, reward, done, truncated, info = env.step(action[0])
    print("Reward : ", reward)
    print("Terminated : ", done)
    return obs_as_tensor(obs)


def obs_as_tensor(obs):
    _obs = {}
    for k, v in obs.items():
        v = torch.tensor(v).unsqueeze(0)
        _obs[k] = v
    return _obs


def save_figure():
    figname = "step-" + str(step) + ".png"
    output_file = os.path.join(input_folder, figname)
    renderer.fig.savefig(output_file)


input_folder = "experiments/hex_env_with_insert/eps-0-05"

checkpoint = os.path.join(input_folder, "best_model.zip")
model = PPO.load(checkpoint)

config_fn = os.path.join(input_folder, "config.yml")
config = load_yaml_config(config_fn)

env = initialize_environment()
obs = obs_as_tensor(env._get_obs())
step = 0

renderer = Renderer(env.graph, env.graph.vertex_coordinates, label_halfedge=True)
renderer.plot()

dist = get_action_distribution(obs)
obs = step_environment(dist)
renderer.plot()
step += 1
# save_figure()
