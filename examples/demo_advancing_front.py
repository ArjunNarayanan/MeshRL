from envs.environment_initializers import Hexagon
from src.render import Renderer
import numpy as np

init = Hexagon(target_angle=60)
mesh, _ = init()

vidx = mesh.insert_half_edge(0, 1)

r = Renderer(mesh, mesh.vertex_coordinates, vertex_size=30, label_vertices=True, label_halfedge=True)