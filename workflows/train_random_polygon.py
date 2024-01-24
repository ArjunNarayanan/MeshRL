from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
import os
import datetime

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.feature_extractor import feature_extractor_initializer
from src.policy import CustomActorCriticPolicy
from src.utils import load_yaml_config


def make_output_dir_if_necessary(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def initialize_environment(env_config, eval=False):
    if eval:
        env_config["incremental_reward"] = False
    env = RandomPolygonEnv.from_config(env_config)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("-num_envs", default=1, type=int)
    parser.add_argument("-config", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    print("TRAIN START TIMESTAMP : ", datetime.datetime.now())

    default_output_dir = os.path.dirname(args.config)
    output_dir = config.get("output_dir", default_output_dir)
    make_output_dir_if_necessary(output_dir)
    print("\n\tUSING OUTPUT DIR : ", output_dir, "\n")

    ###################################################################################################################
    #   SETTING UP ENVIRONMENTS
    print("\n\tINITIALIZING ENVIRONMENT : ")
    env_config = config["environment"]
    if "logdir" not in env_config:
        env_config["logdir"] = output_dir

    num_envs = args.num_envs
    if num_envs > 1:
        env = make_vec_env(
            lambda: initialize_environment(env_config, eval=False),
            num_envs,
            vec_env_cls=SubprocVecEnv,
        )
        eval_env = make_vec_env(
            lambda: initialize_environment(env_config, eval=True),
            num_envs,
            vec_env_cls=SubprocVecEnv,
        )
    else:
        env = initialize_environment()
        eval_env = initialize_environment(env_config, eval=True)
    ###################################################################################################################

    ###################################################################################################################
    # SETTING UP POLICY
    print("\n\tINITIALIZING POLICY : ")
    features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config)
    features_extractor_kwargs.update({"input_features": RandomPolygonEnv.get_feature_size()})

    policy_kwargs = config["policy"]
    policy_kwargs["features_extractor_class"] = features_extractor_class
    policy_kwargs["features_extractor_kwargs"] = features_extractor_kwargs

    ppo_config = config["PPO"]
    ppo_config["policy_kwargs"] = policy_kwargs
    ppo_config["verbose"] = 1
    ppo_config["tensorboard_log"] = output_dir

    model = PPO(
        CustomActorCriticPolicy,
        env,
        **ppo_config,
    )
    ###################################################################################################################

    ###################################################################################################################
    # TRAIN MODEL
    eval_config = config["evaluator"]
    num_evaluations = eval_config["num_evaluations"]
    total_timesteps = config["total_timesteps"]
    eval_freq = int(total_timesteps / (num_evaluations * num_envs))
    print("\n\tEVAL FREQUENCY : ", eval_freq)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        n_eval_episodes=100,
        eval_freq=eval_freq,
        deterministic=False,
        render=False
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    print("TRAIN STOP TIMESTAMP : ", datetime.datetime.now())
    ###################################################################################################################
