from envs.environment_initializers import LEnv
from src.render import Renderer
import os


def save_fig(fig, figname):
    outputfile = os.path.join(root_dir, figname)
    fig.savefig(outputfile)


init = LEnv()
mesh, _ = init()
root_dir = "examples/figures/LEnv"

r = Renderer(mesh, mesh.vertex_coordinates, vertex_size=30)
r.plot()
figname = "initial.pdf"
save_fig(r.fig, figname)

mesh.insert_vertex(0)
r.plot()
figname = "step-1.pdf"
save_fig(r.fig, figname)

mesh.insert_half_edge(3, 3)
r.plot()
figname = "step-2.pdf"
save_fig(r.fig, figname)

mesh.insert_vertex(5)
r.plot()
figname = "step-3.pdf"
save_fig(r.fig, figname)

mesh.insert_half_edge(3, 2)
r.plot()
figname = "step-4.pdf"
save_fig(r.fig, figname)