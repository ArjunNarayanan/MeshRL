import torch
from envs.hex_env_with_insert import HexEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy
from stable_baselines3 import PPO
from src.render import Renderer
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt


def extract_env(wrapped_env):
    env = wrapped_env.envs[0].env
    return env


def plot_distribution(probs):
    fig, ax = plt.subplots()
    ax.bar(range(len(probs)), probs)
    ax.set_xlabel("Actions")
    ax.set_ylabel("Probabilities")
    ax.grid()
    return fig


template_size = 6
env = HexEnv(template_size=template_size)
num_actions = env.template_size * env.num_actions_per_halfedge

policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(
        input_features=5,
        output_features=num_actions,
        number_of_layers=5
    )
)

model = PPO(
    CustomActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    ent_coef=0.005,
    tensorboard_log="examples/hex_env_with_insert_log/"
)
wrapped_env = model.env

eval_env = HexEnv(template_size=template_size)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="examples/hex_env_with_insert_log",
    eval_freq=1000,
    deterministic=True,
    render=False
)

if __name__ == "__main__":
    model.learn(total_timesteps=1000000, callback=eval_callback)

# obs = model.env.reset()
# render_env = extract_env(model.env)
# renderer = Renderer(render_env.graph, coords=render_env.graph.vertex_coordinates)
# renderer.plot()
# save_fig(renderer.fig, "step-3")


# model = PPO.load(
#     "examples/hex_env_with_insert_log/best_model",
#     wrapped_env
# )
#
# render_env = extract_env(model.env)
# renderer = Renderer(render_env.graph, coords=render_env.graph.vertex_coordinates)
# renderer.plot()
#
# obs = obs_as_tensor(obs, torch.device("cpu"))
# with torch.no_grad():
#     dist = model.policy.get_distribution(obs)
# probs = dist.distribution.probs[0]
# fig = plot_distribution(probs)
# action = dist.get_actions()
# obs, reward, done, info = model.env.step(action)
#
# renderer.plot()
