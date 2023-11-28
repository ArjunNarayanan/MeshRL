import torch
from stable_baselines3.common.env_util import make_vec_env
from envs.hex_env_with_insert import HexEnv
from src.feature_extractor import FeatureExtractor
import timeit

vec_env = make_vec_env(HexEnv, 64)

obs = vec_env.reset()


twin = torch.tensor(obs["twin"])

t1 = FeatureExtractor._archive_unroll_and_offset_indices(twin)
t2 = FeatureExtractor.unroll_and_offset_indices(twin)

setup = """
from src.feature_extractor import FeatureExtractor
import torch
from stable_baselines3.common.env_util import make_vec_env
from envs.hex_env_with_insert import HexEnv

vec_env = make_vec_env(HexEnv, 64)
obs = vec_env.reset()
twin = torch.tensor(obs["twin"])
"""


timeit.timeit(
    "FeatureExtractor._archive_unroll_and_offset_indices(twin)",
    setup=setup,
    number=1000
)
timeit.timeit(
    "FeatureExtractor.unroll_and_offset_indices(twin)",
    setup=setup,
    number=1000
)