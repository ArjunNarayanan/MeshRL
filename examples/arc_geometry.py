import numpy as np
from envs.environment_initializers import Arc
from src.render import Renderer


init = Arc()
g, d = init()

r = Renderer(g, g.vertex_coordinates, label_vertices=True, vertex_size=30, fontsize=20)
r.plot()
r.fig.tight_layout()
filename = "examples/figures/arc.pdf"
r.fig.savefig(filename)