import numpy as np
from envs.environment_initializers import SquareHole2
from src.render import Renderer

init = SquareHole2()
g, d = init()

r = Renderer(g, g.vertex_coordinates, label_vertices=True, vertex_size=30, fontsize=20)
r.plot()
r.fig.tight_layout()
filename = "examples/figures/square-hole.pdf"
r.fig.savefig(filename)
