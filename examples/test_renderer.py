from src.polygraph import PolyGraph
from src.render import Renderer
import numpy as np
import matplotlib.pyplot as plt


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(6), coords))
    return coords


coords = generate_coordinates()
graph = PolyGraph.from_face_loops([[0, 1, 2, 3, 4, 5]], vertex_coordinates=coords)
renderer = Renderer(graph, graph.vertex_coordinates)
renderer.plot()

graph.insert_edge(0, 2)
renderer.plot()

graph.insert_vertex(7)
renderer.plot()

graph.insert_edge(8, 1)
renderer.plot()

graph.insert_edge(5, 1)
renderer.plot()

graph.insert_edge(6, 1)
renderer.plot()

graph.insert_edge(15, 1)
renderer.plot()