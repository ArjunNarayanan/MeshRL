from envs.hex_env_with_insert import HexEnv
from src.render import Renderer
import numpy as np






env = HexEnv()
renderer = Renderer(env.graph, env.graph.vertex_coordinates)
renderer.plot()

env.step(0)
renderer.plot()

env.step(4)