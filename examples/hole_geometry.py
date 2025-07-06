from envs.environment_initializers import SquareHole, SquareHole2
from src.render import Renderer

init = SquareHole(90)
g, d = init()

r = Renderer(g, g.vertex_coordinates, label_vertices=True, vertex_size=30, fontsize=20)
r.plot()
r.fig.tight_layout()
filename = "examples/figures/square-hole.pdf"
r.fig.savefig(filename)

init = SquareHole2()
g, d = init()

r = Renderer(g, g.vertex_coordinates, label_vertices=True, vertex_size=30, fontsize=20)
r.plot()
r.fig.tight_layout()
filename = "examples/figures/square-hole2.pdf"
r.fig.savefig(filename)