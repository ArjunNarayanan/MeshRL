from envs.environment_initializers import Hexagon
from src.render import Renderer

init = Hexagon(target_angle=np.pi / 3)
mesh, _ = init()

mesh.insert_half_edge(0, 2)
mesh.insert_vertex(6)
mesh.insert_half_edge(1, 2)
mesh.insert_half_edge(2, 1)
mesh.insert_half_edge(4, 2)
mesh.insert_half_edge(5, 1)

r = Renderer(mesh, mesh.vertex_coordinates, vertex_size=30)
r.plot()
r.fig.tight_layout()
outputfile = "examples/figures/operations/perfect_hexagon.pdf"
r.fig.savefig(outputfile)
