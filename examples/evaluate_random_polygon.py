import torch
from stable_baselines3 import PPO
from envs.random_polygon_env import RandomPolygonEnv
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
    env = RandomPolygonEnv.from_config(env_config)
    return env


def get_action_distribution(obs, plot=False):
    template_halfedges = env.index_to_halfedge
    print("Template : ", template_halfedges)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
    probs = dist.distribution.probs[0]
    if plot:
        plot_distribution(probs)
    return dist


def step_environment(dist):
    action = dist.get_actions()
    action = action.cpu().numpy()
    linear_action = action[0]
    action_probability = dist.distribution.probs[0, linear_action].item()

    action_halfedge, action_type = env._linear_action_index_to_halfedge_and_action(action[0])

    print("Step : ", env.num_steps)
    print("Selected action : ", action[0])
    print("Selected action probability : ", action_probability)
    print("\tHalfedge: ", action_halfedge, " \tType: ", action_type)
    obs, reward, done, truncated, info = env.step(action[0])
    print("Reward : ", reward)
    print("Terminated : ", done)

    return obs, done


def obs_as_tensor(obs):
    _obs = {}
    for k, v in obs.items():
        v = torch.tensor(v).unsqueeze(0)
        _obs[k] = v
    return _obs


def save_figure():
    figname = "step-" + str(step) + ".png"
    output_file = os.path.join(fig_output_folder, figname)
    renderer.fig.savefig(output_file)


def make_output_folder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def reset_if_done(obs, done):
    if done:
        total_reward = abs(env.initial_score - env.score)
        print("\nTOTAL REWARD : ", total_reward, "\n")
        env.reset()
        renderer.graph = env.graph
        renderer.coords = env.graph.vertex_coordinates
        obs = obs_as_tensor(env._get_obs())
        return obs
    else:
        return obs_as_tensor(obs)


def plot_env():
    renderer.plot()
    renderer.plot_vertex_scores(env.vertex_desired_degree)


input_folder = "experiments/random-polygon/tri-poly-10"
fig_output_folder = os.path.join(input_folder, "figures")
make_output_folder(fig_output_folder)

checkpoint = os.path.join(input_folder, "best_model.zip")
model = PPO.load(checkpoint)

config_fn = os.path.join(input_folder, "config.yml")
config = load_yaml_config(config_fn)

env = initialize_environment()
obs = obs_as_tensor(env._get_obs())
renderer = Renderer(env.graph, env.graph.vertex_coordinates)
plot_env()

step = 0
dist = get_action_distribution(obs)
obs, done = step_environment(dist)
plot_env()
obs = reset_if_done(obs, done)
step += 1

# save_figure()
