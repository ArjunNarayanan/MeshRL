from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from envs.hex_env_with_insert import HexEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy
from src.render import Renderer
# import matplotlib.pyplot as plt


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


def make_output_dir_if_necessary(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


template_size = 18
num_actions_per_halfedge = 6
num_envs = 6
# env = HexEnv(template_size=template_size)
env = make_vec_env(HexEnv, num_envs)


num_actions = template_size * num_actions_per_halfedge
output_dir = "/home/anarayan/Research/MeshRL/Polygraph/experiments/hex_env_with_insert"
make_output_dir_if_necessary(output_dir)

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
    ent_coef=0.01,
    tensorboard_log=output_dir
)
wrapped_env = model.env

eval_env = HexEnv(template_size=template_size)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=output_dir,
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
