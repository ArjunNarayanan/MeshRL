from envs.environment_initializers import FixedRandomPolygon
from src.render import Renderer


rpoly = FixedRandomPolygon(10, 90)
graph, vd = rpoly()

render = Renderer(graph, graph.vertex_coordinates, label_halfedge=True)
render.plot()

graph.insert_half_edge(6,3)
graph.insert_vertex(11)
graph.insert_half_edge(12, 3)