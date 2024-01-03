import torch
from envs.hex_env import HexEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy
from stable_baselines3 import PPO
from src.render import Renderer
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
import matplotlib.pyplot as plt


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(6), coords))
    return coords


def extract_env(wrapped_env):
    env = wrapped_env.envs[0].env
    return env


env = HexEnv()
coords = generate_coordinates()

policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(
        input_features=4,
        output_features=16,
        number_of_layers=5
    )
)

model = PPO(
    CustomActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    ent_coef=0.005,
    tensorboard_log="examples/hex_env_log/"
)


#
# model.learn(total_timesteps=10000)

def plot_distribution(probs):
    fig, ax = plt.subplots()
    ax.bar(range(len(probs)), probs)
    ax.set_xlabel("Actions")
    ax.set_ylabel("Probabilities")
    ax.grid()
    return fig


def save_fig(fig, filename):
    import os
    filepath = os.path.join("examples", "hex_env_log", "figures", filename + ".png")
    fig.savefig(filepath)

obs = model.env.reset()
render_env = extract_env(model.env)
renderer = Renderer(render_env.graph, coords=coords)
renderer.plot()
save_fig(renderer.fig, "step-3")

obs = obs_as_tensor(obs, torch.device("cpu"))
with torch.no_grad():
    dist = model.policy.get_distribution(obs)
probs = dist.distribution.probs[0]
fig = plot_distribution(probs)
action = dist.get_actions()
obs, reward, done, info = model.env.step(action)

renderer.plot()
