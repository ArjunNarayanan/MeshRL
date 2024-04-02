from envs.substep_angle_env import AngleEnv as SubstepAngleEnv
from envs.random_polygon_tiler_env import RandomPolygonEnv
from envs.global_angle_env import AngleEnv as GlobalAngleEnv
from envs.angle_env_with_length import AngleEnvWithLength
import envs.environment_initializers as init


def get_env_initializer(init_config):
    name = init_config["name"]
    angle = init_config["target_angle"]
    if name == "LEnv":
        return init.LEnv(angle)
    elif name == "Hexagon":
        return init.Hexagon(angle)
    elif name == "CenterCrack":
        return init.CenterCrack(angle)
    elif name == "SquareHole":
        return init.SquareHole(angle)
    elif name == "RandomPolygon":
        min_degree = init_config["min_polygon_degree"]
        max_degree = init_config["max_polygon_degree"]
        scale = init_config.get("scale", 0.5)
        degree_range = list(range(min_degree, max_degree + 1))
        return init.RandomPolygon(degree_range, angle, scale=scale)
    else:
        raise ValueError("Unexpected env initializer with name : ", name)


def initialize_environment(env_config):
    env_config = env_config.copy()

    env_name = env_config["name"]
    initializer = get_env_initializer(env_config["initializer"])
    env_config["graph_initializer"] = initializer

    if env_name == "RandomPolygonEnv":
        return RandomPolygonEnv.from_config(env_config)
    elif env_name == "SubstepAngleEnv":
        return SubstepAngleEnv.from_config(env_config)
    elif env_name == "GlobalAngleEnv":
        return GlobalAngleEnv.from_config(env_config)
    elif env_name == "AngleEnvWithLength":
        return AngleEnvWithLength.from_config(env_config)
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
    elif env_name == "AngleEnvWithLength":
        return AngleEnvWithLength.get_feature_size()
    else:
        raise TypeError("Unexpected environment name : ", env_name)
