import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.getcwd())
from envs.random_polygon_env import RandomPolygonEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy
from src.utils import load_yaml_config

def initialize_environment():
    env_config = config["environment"]
    env = RandomPolygonEnv.from_config(env_config)
    return env


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.5, log=True)

    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)

    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 11)
    batch_size = 2 ** trial.suggest_int("batch_size", 5, 9)

    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    feature_extractor_layers = trial.suggest_int("feature_extractor_layers", 2, 10)
    feature_extractor_size = 2 ** trial.suggest_int("feature_extractor_size", 5, 10)

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(
            input_features=RandomPolygonEnv.get_feature_size(),
            output_features=feature_extractor_size,
            number_of_layers=feature_extractor_layers
        ),
        ortho_init=ortho_init,
    )

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": policy_kwargs,
    }


class Objective:
    def __init__(self, gpu_queue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        env = make_vec_env(initialize_environment, NUM_ENVS)
        DEFAULT_HYPERPARAMS = {
            "policy": CustomActorCriticPolicy,
            "env": env,
            "verbose": 1,
        }