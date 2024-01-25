import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage, JournalFileStorage
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import os
import sys
import datetime

sys.path.append(os.getcwd())
from envs.random_polygon_tiler_env import initialize_environment, RandomPolygonEnv
from src.transformer_feature_extractor import TransformerFeatureExtractor as FeatureExtractor
from src.policy import CustomActorCriticPolicy
from src.utils import load_yaml_config


def sample_ppo_params(trial: optuna.Trial):
    """Sampler for PPO hyperparameters."""
    gamma = 1.0
    n_steps = 1024
    batch_size = 256

    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.5, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.00000001, 0.5, log=True)
    clip_range = trial.suggest_float("clip_range", 0.01, 0.20)

    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 5.0, log=True)
    learning_rate = trial.suggest_float("lr", 1e-6, 0.001, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])

    feature_extractor_layers = trial.suggest_int("feature_extractor_layers", 2, 10)
    num_features_per_head = 2 ** trial.suggest_int("feature_extractor_size", 4, 7)
    num_heads = 2 * trial.suggest_int("num_heads", 2, 6)
    feature_extractor_size = num_features_per_head * num_heads
    dropout = trial.suggest_float("dropout", 1e-3, 0.4, log=True)

    sequence_length = config["environment"]["template_size"]

    feature_extractor_kwargs = dict(
        input_features=RandomPolygonEnv.get_feature_size(),
        output_features=feature_extractor_size,
        sequence_length=sequence_length,
        num_heads=num_heads,
        number_of_layers=feature_extractor_layers,
        dropout=dropout
    )

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=feature_extractor_kwargs,
        ortho_init=ortho_init,
    )

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": policy_kwargs,
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
            self,
            eval_env,
            trial: optuna.Trial,
            eval_freq,
            n_eval_episodes: int = 100,
            deterministic: bool = False,
            verbose: int = 0,
            best_model_save_path=None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class Objective:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id

    def __call__(self, trial):

        env = make_vec_env(
            lambda: initialize_environment(env_config),
            NUM_ENVS,
            vec_env_cls=SubprocVecEnv
        )

        trial_logdir = os.path.join(output_folder, "trial-" + str(trial.number))
        DEFAULT_HYPERPARAMS = {
            "policy": CustomActorCriticPolicy,
            "env": env,
            "verbose": 1,
        }

        kwargs = DEFAULT_HYPERPARAMS.copy()
        kwargs.update(sample_ppo_params(trial))
        kwargs["device"] = torch.device("cuda:" + str(self.gpu_id))
        kwargs["tensorboard_log"] = trial_logdir

        model = PPO(**kwargs)

        eval_env = make_vec_env(
            lambda: initialize_environment(env_config, eval=True),
            NUM_ENVS
        )

        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=False,
            verbose=1,
        )

        nan_encountered = False
        try:
            model.learn(N_TIMESTEPS, callback=eval_callback)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)
            nan_encountered = True
        finally:
            # Free memory.
            model.env.close()
            eval_env.close()

        # Tell the optimizer that the trial failed.
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize PPO hyperparams with Optuna")
    parser.add_argument("-num_envs", default=10, type=int)
    parser.add_argument("-config", required=True)
    parser.add_argument("-gpu", default=0, type=int)
    args = parser.parse_args()

    NUM_ENVS = args.num_envs
    config_filename = args.config
    gpu_id = args.gpu
    config = load_yaml_config(config_filename)
    env_config = config["environment"]

    N_TRIALS = int(config["num_trials"])
    N_STARTUP_TRIALS = 5
    N_EVALUATIONS = 500
    N_EVAL_EPISODES = 100
    N_TIMESTEPS = int(config["total_timesteps"])

    print("EXPERIMENT START TIMESTAMP : ", datetime.datetime.now(), "\n\n")

    print("\nTotal timesteps : ", N_TIMESTEPS, "\n")
    EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
    print("\nEval Freq : ", EVAL_FREQ, "\n")
    JOBID = os.environ.get("SLURM_JOB_ID")

    # Set pytorch num threads to 1 for faster training.
    # This seems to be slower when we use > 1 GPUs and SubprocVecEnv
    # torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study_name = config["study_name"]

    output_folder = os.path.dirname(config_filename)
    journal_file_name = study_name + ".log"
    storage_path = os.path.join(output_folder, journal_file_name)

    storage = JournalStorage(JournalFileStorage(storage_path))
    print("\nUsing storage : ", storage_path)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )

    try:
        study.optimize(
            Objective(gpu_id),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    print("EXPERIMENT END TIMESTAMP : ", datetime.datetime.now(), "\n\n")
