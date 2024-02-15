import pickle
import os
import sys

sys.path.append(os.getcwd())
from src.tiler import Tiler
from src.render import Renderer
from envs.angle_env import AngleEnv
from src.utils import load_yaml_config

if __name__ == "__main__":
    config_fn = "../experiments/angle-env/quad/face-angle-vertex/config.yml"
    config = load_yaml_config(config_fn)

    env_config = config["environment"]
    env = AngleEnv.from_config(env_config)

    env.step(0)
