from envs.random_polygon_env import RandomPolygonEnv
import timeit


setup = """
from envs.random_polygon_env import RandomPolygonEnv

def operate(env):
    env._select_halfedge_template_center(env.graph.halfedge_list())
    env._build_template()
    obs = env._get_obs()

env = RandomPolygonEnv(3, [40], 20, 3)
"""

timeit.timeit("operate(env)", setup=setup, number=1000)