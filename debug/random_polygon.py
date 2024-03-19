from envs.environment_initializers import RandomPolygon
from src.render import Renderer


rpoly = RandomPolygon([10], 90)
graph, vd = rpoly()

render = Renderer(graph, graph.vertex_coordinates)
render.plot()