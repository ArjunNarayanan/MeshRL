import numpy as np
from src.tiler import Tiler
from src.render import Renderer


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    return coords


def initialize_graph():
    coords = generate_coordinates()
    coords = dict(zip(range(6), coords))
    graph = Tiler.from_face_loops(
        [[0, 1, 2, 3, 4, 5]],
        vertex_coordinates=coords
    )
    return graph


graph = initialize_graph()
renderer = Renderer(
    graph,
    coords=graph.vertex_coordinates,
    label_vertices=True,
)
renderer.plot()

graph.insert_half_edge(0, 2)
renderer.plot()


