from stable_baselines3.common.env_util import make_vec_env
from envs.hex_env_with_insert import HexEnv

vec_env = make_vec_env(HexEnv, 4)