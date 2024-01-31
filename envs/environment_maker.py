from envs.angle_env import AngleEnv
from envs.random_polygon_tiler_env import RandomPolygonEnv


def initialize_environment(env_config):
    env_name = env_config["name"]
    if env_name == "RandomPolygonEnv":
        return RandomPolygonEnv.from_config(env_config)
    elif env_name == "AngleEnv":
        return AngleEnv.from_config(env_config)
    else:
        raise TypeError("Unexpected environment name : ", env_name)


def get_env_feature_size(env_config):
    env_name = env_config["name"]
    if env_name == "RandomPolygonEnv":
        return RandomPolygonEnv.get_feature_size()
    elif env_name == "AngleEnv":
        return AngleEnv.get_feature_size()
    else:
        raise TypeError("Unexpected environment name : ", env_name)
