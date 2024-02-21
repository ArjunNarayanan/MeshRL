from envs.substep_angle_env import AngleEnv as SubstepAngleEnv
from envs.random_polygon_tiler_env import RandomPolygonEnv
from envs.global_angle_env import AngleEnv as GlobalAngleEnv


def initialize_environment(env_config):
    env_name = env_config["name"]
    if env_name == "RandomPolygonEnv":
        return RandomPolygonEnv.from_config(env_config)
    elif env_name == "SubstepAngleEnv":
        return SubstepAngleEnv.from_config(env_config)
    elif env_name == "GlobalAngleEnv":
        return GlobalAngleEnv.from_config(env_config)
    else:
        raise TypeError("Unexpected environment name : ", env_name)


def get_env_feature_size(env_config):
    env_name = env_config["name"]
    if env_name == "RandomPolygonEnv":
        return RandomPolygonEnv.get_feature_size()
    elif env_name == "SubstepAngleEnv":
        return SubstepAngleEnv.get_feature_size()
    elif env_name == "GlobalAngleEnv":
        return GlobalAngleEnv.get_feature_size()
    else:
        raise TypeError("Unexpected environment name : ", env_name)
