import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.hex_env_with_insert import HexEnv
from src.render import Renderer
from stable_baselines3.common.utils import obs_as_tensor
import matplotlib.pyplot as plt


def plot_distribution(probs):
    fig, ax = plt.subplots()
    xtick_locs = range(len(probs))

    ax.bar(xtick_locs, probs)
    # ax.set_xticks(xtick_locs)
    ax.set_xlabel("Actions")
    ax.set_ylabel("Probabilities")
    ax.grid()
    return fig


def initialize_env():
    template_size = 12
    max_actions = 12
    no_action_reward = -4

    env = HexEnv(
        template_size=template_size,
        no_action_reward=no_action_reward,
        max_actions=max_actions,
        incremental_reward=False
    )
    return env


checkpoint = "experiments/hex_env_with_insert/best_model.zip"
model = PPO.load(checkpoint)
env = make_vec_env(initialize_env, 1)

extracted_env = env.envs[0].env
renderer = Renderer(extracted_env.graph, extracted_env.graph.vertex_coordinates)
renderer.plot()

obs = env.reset()
obs = obs_as_tensor(obs, torch.device("cpu"))
with torch.no_grad():
    dist = model.policy.get_distribution(obs)
probs = dist.distribution.probs[0]
fig = plot_distribution(probs)

action = dist.get_actions()
action = action.cpu().numpy()
print("Selected action : ", action)
obs, reward, done, info = env.step(action)
print("Reward : ", reward)

extracted_env = env.envs[0].env
renderer = Renderer(extracted_env.graph, extracted_env.graph.vertex_coordinates)
renderer.plot()
