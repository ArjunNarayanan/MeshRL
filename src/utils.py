import yaml
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.feature_extractor import feature_extractor_initializer
from stable_baselines3 import PPO


def load_yaml_config(config_fn):
    print("\nLOADING CONFIG FILE AT : ", config_fn)
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def load_model_from_checkpoint(checkpoint_file, config_file):
    config = load_yaml_config(config_file)
    features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config)
    features_extractor_kwargs.update({"input_features": RandomPolygonEnv.get_feature_size()})
    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
    )
    ppo_config = dict(policy_kwargs=policy_kwargs)
    model = PPO.load(checkpoint_file, custom_objects=ppo_config)
    return model
