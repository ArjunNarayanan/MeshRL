from stable_baselines3.common.callbacks import EventCallback
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from copy import deepcopy
from stable_baselines3.common.env_util import make_vec_env
from envs.environment_maker import initialize_environment
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Any, Dict


class AverageBestCallback(EventCallback):
    def __init__(
            self,
            env_config,
            eval_freq,
            best_model_save_path,
            vec_env_class=None,
            trials_per_env=10,
            trials_per_eval=20,
            num_envs=10,
            deterministic=False,
            warn=True
    ):
        super().__init__(None, verbose=1)
        self.callback_on_new_best = None
        self.env_config = env_config
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        self.trials_per_env = trials_per_env
        self.trials_per_eval = trials_per_eval
        self.num_envs = num_envs
        self.deterministic = deterministic
        self.warn = warn

        if vec_env_class is None:
            vec_env_class = DummyVecEnv
        self.vec_env_class = vec_env_class

    def _init_callback(self) -> None:
        os.makedirs(self.best_model_save_path, exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _initialize_eval_env(self):
        env_config = deepcopy(self.env_config)
        init = env_config["initializer"]
        init["name"] = "FixedRandomPolygon"

        min_polygon_degree = init["min_polygon_degree"]
        max_polygon_degree = init["max_polygon_degree"]
        polygon_degree = np.random.choice(range(min_polygon_degree, max_polygon_degree + 1))
        init["polygon_degree"] = polygon_degree

        eval_env = make_vec_env(
            lambda: initialize_environment(env_config),
            self.num_envs,
            vec_env_cls=self.vec_env_class
        )
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        continue_training = True
        if not (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            # do not run evaluation in this scenario
            return continue_training

        # Reset success rate buffer
        self._is_success_buffer = []
        episode_rewards = []
        episode_lengths = []

        for step in range(self.trials_per_eval):
            self._initialize_eval_env()
            trial_rewards, trial_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.trials_per_env,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            max_idx = np.argmax(trial_rewards)
            episode_rewards.append(trial_rewards[max_idx])
            episode_lengths.append(trial_lengths[max_idx])

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward

        if self.verbose >= 1:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
