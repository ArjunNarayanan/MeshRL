from envs.hex_env import HexEnv
from src.feature_extractor import FeatureExtractor
from src.policy import CustomActorCriticPolicy
from stable_baselines3 import PPO

env = HexEnv()
policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(
        input_features=4,
        output_features=16,
        number_of_layers=5
    )
)

if __name__=="__main__":
    model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    total_timesteps, callback = model._setup_learn(10)
    callback.on_training_start(locals(), globals())

    continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

    print("Num samples : ", len(model.rollout_buffer))
