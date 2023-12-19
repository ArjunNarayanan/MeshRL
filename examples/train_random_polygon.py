from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import argparse
from stable_baselines3.common.env_util import make_vec_env
import sys
import os

sys.path.append(os.getcwd())
from envs.random_polygon_env import RandomPolygonEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy
from src.utils import load_yaml_config


def extract_env(wrapped_env):
    env = wrapped_env.envs[0].env
    return env


def make_output_dir_if_necessary(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def initialize_environment():
    env_config = config["environment"]
    env = RandomPolygonEnv.from_config(env_config)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("-num_envs", default=1, type=int)
    parser.add_argument("-config", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    max_edge_addition_steps = config["environment"].get("max_edge_addition_steps", 3)

    num_envs = args.num_envs
    if num_envs > 1:
        env = make_vec_env(initialize_environment, num_envs)
    else:
        env = initialize_environment()

    template_size = config["environment"]["template_size"]
    num_actions = template_size * RandomPolygonEnv.get_num_actions_per_halfedge(max_edge_addition_steps)

    default_output_dir = os.path.dirname(args.config)
    output_dir = config.get("output_dir", default_output_dir)
    make_output_dir_if_necessary(output_dir)
    print("\n\tUSING OUTPUT DIR: ", output_dir, "\n")


    feature_extractor_layers = config["feature_extractor"]["number_of_layers"]
    feature_extractor_size = config["feature_extractor"]["output_features"]
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(
            input_features=RandomPolygonEnv.get_feature_size(),
            output_features=feature_extractor_size,
            number_of_layers=feature_extractor_layers
        ),
        ortho_init=config["feature_extractor"]["ortho_init"]
    )

    ppo_config = config["PPO"]
    ppo_config["policy_kwargs"] = policy_kwargs
    ppo_config["verbose"] = 1
    ppo_config["tensorboard_log"] = output_dir    

    model = PPO(
        CustomActorCriticPolicy,
        env,
        **ppo_config,
    )

    eval_env = make_vec_env(initialize_environment, 1)

    eval_config = config["evaluator"]
    eval_freq = eval_config["eval_freq"]
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        n_eval_episodes=100,
        eval_freq=eval_freq,
        deterministic=False,
        render=False
    )

    total_timesteps = config["total_timesteps"]
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
