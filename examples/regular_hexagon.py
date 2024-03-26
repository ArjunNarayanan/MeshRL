from envs.environment_initializers import Hexagon
from src.render import Renderer

init = Hexagon()
mesh, _ = init()

mesh.insert_half_edge(0, 2)
mesh.insert_vertex(6)

r = Renderer(mesh, mesh.vertex_coordinates, label_halfedge=True)
r.plot()
