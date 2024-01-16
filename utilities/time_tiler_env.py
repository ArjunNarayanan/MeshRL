from envs.random_polygon_tiler_env import RandomPolygonEnv
import timeit


setup = """
from envs.random_polygon_tiler_env import RandomPolygonEnv

env = RandomPolygonEnv(3, [40], 50)
"""

timeit.timeit("env._get_feature_matrix()", setup=setup, number=1000)