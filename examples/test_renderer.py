from envs.regular_polygon_env import RegularPolygonEnv
from src.render import Renderer
import numpy as np

env = RegularPolygonEnv(
    8,
    10,
    3,
    2,
    True,
    "",
)
renderer = Renderer(env.graph, env.graph.vertex_coordinates)
renderer.plot()

# obs1 = env.step(31)
# renderer.plot()
#
# obs2 = env.step(40)
# renderer.plot()
#
# obs3 = env.step(19)
# renderer.plot()
#
# obs4 = env.step(24)
# renderer.plot()
#
# obs5 = env.step(1)
# renderer.plot()
#
# obs6 = env.step(6)
# renderer.plot()
