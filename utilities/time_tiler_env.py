from envs.random_polygon_tiler_env import RandomPolygonEnv
import timeit


setup = """
from envs.random_polygon_tiler_env import RandomPolygonEnv

def operate(env):
    env._select_half_edge_template_center(env.graph.half_edge_list())
    env._build_template()
    obs = env._get_obs()

env = RandomPolygonEnv(3, [40])
"""

timeit.timeit("operate(env)", setup=setup, number=1000)