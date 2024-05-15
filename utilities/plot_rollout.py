import torch
import os
import sys
import argparse
import math
import glob

sys.path.append(os.getcwd())
from envs.environment_maker import initialize_environment
from src.render import Renderer
from src.utils import load_yaml_config, load_model_from_checkpoint


def get_action_distribution(obs):
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)

    return dist


def step_environment(env, dist):
    action = dist.get_actions()
    action = action.cpu().numpy()

    print("Step : ", env.num_steps)
    obs, reward, done, truncated, info = env.step(action[0])
    print("Reward : ", reward)

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


def plot_env(env, filename=None, plot_vertex_scores=False):
    renderer = Renderer(env.graph, env.graph.vertex_coordinates)
    renderer.plot()
    if plot_vertex_scores:
        renderer.plot_vertex_scores(env.vertex_desired_degree)
    renderer.plot_face_scores(env.face_desired_degree)
    if filename is not None:
        renderer.fig.savefig(filename)


def plot_rollout(rollout, plot_vertex_scores=False):
    fig_output_folder = os.path.join(output_folder, "rollout-" + str(rollout))
    make_output_folder(fig_output_folder)

    step = 0
    figname = "step-" + str(step).zfill(NUM_DIGITS) + ".png"
    output_file = os.path.join(fig_output_folder, figname)
    plot_env(env, filename=output_file, plot_vertex_scores=plot_vertex_scores)
    obs = obs_as_tensor(env._get_obs())
    done = env.is_terminated()

    while not done:
        step += 1
        dist = get_action_distribution(obs)
        obs, done = step_environment(env, dist)
        obs = obs_as_tensor(obs)

        env.graph.smooth_vertices(num_iter=smooth_iterations)

        figname = "step-" + str(step).zfill(NUM_DIGITS) + ".png"
        output_file = os.path.join(fig_output_folder, figname)

        plot_env(env, filename=output_file, plot_vertex_scores=plot_vertex_scores)
        print("Saving figure : ", output_file)

    total_reward = abs(env.initial_score - env.score) / env.initial_score
    print("\nNORMALIZED RETURN : ", total_reward, "\n")


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
    parser.add_argument("-model", default="best_model.zip")
    parser.add_argument("-rollout", default=None)
    parser.add_argument("-scores", default=False, type=bool)
    parser.add_argument("-smooth", default=0, type=int)
    parser.add_argument("-config", default="config.yml")

    args = parser.parse_args()
    input_folder = args.input
    plot_vertex_scores = args.scores
    smooth_iterations = args.smooth

    output_folder = os.path.join(input_folder, "figures")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if not args.rollout:
        rollout_index = get_next_rollout_index()
    else:
        rollout_index = args.rollout

    config_fn = os.path.join(input_folder, args.config)
    config = load_yaml_config(config_fn)

    checkpoint = os.path.join(input_folder, args.model)
    model = load_model_from_checkpoint(checkpoint, config_fn)

    env_config = config["environment"]

    env = initialize_environment(env_config)
    max_steps = env.max_steps
    NUM_DIGITS = math.ceil(math.log10(max_steps))

    print(NUM_DIGITS)

    plot_rollout(rollout_index, plot_vertex_scores=plot_vertex_scores)
