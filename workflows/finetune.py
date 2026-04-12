"""
finetune.py — load an existing PPO checkpoint and continue training.

Usage:
    python workflows/finetune.py \
        -config  experiments/angle-env-with-length/quad/quad-curriculum-phase2/config.yml \
        -checkpoint experiments/angle-env-with-length/quad/quad-curriculum-phase1/best_model.zip \
        -num_envs 8
"""

import argparse
import datetime
import os
import sys

sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from envs.environment_maker import initialize_environment, get_env_feature_size
from src.feature_extractor import feature_extractor_initializer
from src.policy import CustomActorCriticPolicy
from src.utils import load_yaml_config


def make_output_dir(path):
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a PPO checkpoint")
    parser.add_argument("-config",     required=True,  help="Path to config YAML")
    parser.add_argument("-checkpoint", required=True,  help="Path to starting best_model.zip")
    parser.add_argument("-num_envs",   default=4, type=int, help="Parallel envs")
    args = parser.parse_args()

    config     = load_yaml_config(args.config)
    checkpoint = args.checkpoint
    num_envs   = args.num_envs

    print("FINETUNE START : ", datetime.datetime.now())
    print("  checkpoint   : ", checkpoint)
    print("  config       : ", args.config)
    print("  num_envs     : ", num_envs)

    output_dir = config.get("output_dir", os.path.dirname(args.config))
    make_output_dir(output_dir)
    print("  output_dir   : ", output_dir)

    # ── Environments ─────────────────────────────────────────────────────────
    env_config          = config["environment"]
    env_config["logdir"] = output_dir
    eval_env_config     = env_config.copy()

    VecEnvCls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    env      = make_vec_env(lambda: initialize_environment(env_config),      num_envs, vec_env_cls=VecEnvCls)
    eval_env = make_vec_env(lambda: initialize_environment(eval_env_config), num_envs, vec_env_cls=VecEnvCls)

    # ── Policy kwargs (must match the checkpoint's architecture) ─────────────
    features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config)
    features_extractor_kwargs["input_features"] = get_env_feature_size(env_config)
    policy_kwargs = {
        **config["policy"],
        "features_extractor_class":  features_extractor_class,
        "features_extractor_kwargs": features_extractor_kwargs,
    }

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("\nLoading checkpoint…")
    model = PPO.load(
        checkpoint,
        env=env,
        custom_objects={"policy_kwargs": policy_kwargs},
        verbose=1,
        tensorboard_log=output_dir,
    )
    # Override hyperparameters with the new config values
    ppo_cfg = config["PPO"]
    model.learning_rate = ppo_cfg["learning_rate"]
    model.ent_coef      = ppo_cfg["ent_coef"]
    model.clip_range    = ppo_cfg["clip_range"]
    model.gae_lambda    = ppo_cfg["gae_lambda"]
    model.max_grad_norm = ppo_cfg["max_grad_norm"]
    model.vf_coef       = ppo_cfg["vf_coef"]
    print("Checkpoint loaded. Continuing training with updated hyperparameters.")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    total_timesteps = config["total_timesteps"]
    eval_freq       = int(total_timesteps / (config["evaluator"]["num_evaluations"] * num_envs))
    eval_callback   = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        n_eval_episodes=100,
        eval_freq=eval_freq,
        deterministic=False,
        render=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        reset_num_timesteps=False,   # continue step counter from checkpoint
    )

    print("FINETUNE STOP : ", datetime.datetime.now())
