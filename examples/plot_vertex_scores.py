from envs.regular_polygon_env import RegularPolygonEnv
from src.render import Renderer

env = RegularPolygonEnv(
    10,
    12,
    10,
    0,
    True,
    ""
)
renderer = Renderer(env.graph, env.graph.vertex_coordinates, label_halfedge=False, label_vertices=False)
renderer.plot()
renderer.plot_vertex_scores(env.vertex_desired_degree)
