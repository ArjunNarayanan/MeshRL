import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.hex_env_with_insert import HexEnv
from src.render import Renderer
from stable_baselines3.common.utils import obs_as_tensor
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


input_folder = "experiments/hex_env_with_insert/eps-0-05"

checkpoint = os.path.join(input_folder, "best_model.zip")
model = PPO.load(checkpoint)

config_fn = os.path.join(input_folder, "config.yml")
config = load_yaml_config(config_fn)

env = make_vec_env(initialize_environment, 1)

extracted_env = env.envs[0].env
renderer = Renderer(extracted_env.graph, extracted_env.graph.vertex_coordinates)
renderer.plot()

obs, reward, done, _, _ = extracted_env.step(30)
renderer.plot()

# obs = env.reset()
# obs = obs_as_tensor(obs, torch.device("cpu"))
# with torch.no_grad():
#     dist = model.policy.get_distribution(obs)
# probs = dist.distribution.probs[0]
# fig = plot_distribution(probs)
#
# action = dist.get_actions()
# action = action.cpu().numpy()
# action_halfedge = action[0]//6
# action_type = action[0]%6
# print("Selected action : ", action[0])
# print("\tHalfedge: ", action_halfedge, " \tType: ", action_type)
# obs, reward, done, info = env.step(action)
# print("Reward : ", reward)
#
# extracted_env = env.envs[0].env
# template_halfedges = extracted_env.index_to_halfedge
# print("Template : ", template_halfedges)
# renderer = Renderer(extracted_env.graph, extracted_env.graph.vertex_coordinates)
# renderer.plot()
