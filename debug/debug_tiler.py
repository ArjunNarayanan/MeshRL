import numpy as np
from src.tiler import Tiler
from src.render import Renderer


def initialize_graph():
    face_loops = [
        [0, 1, 3, 5, 4],
        [1, 2, 7, 6, 3],
        [5, 3, 6, 8],
        [4, 5, 8, 10, 9],
        [6, 7, 11, 10, 8]
    ]
    vertex_coordinates = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, vertex_coordinates)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [0.1, 0],
        [1, 0],
        [0.7, 0.1],
        [0, 0.8],
        [0.6, 0.4],
        [0.9, 0.3],
        [1, 0.8],
        [0.7, 0.2],
        [0, 1],
        [0.9, 1],
        [1, 1]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


graph = initialize_graph()
vertices = [1, 3, 4, 5, 6, 7, 8, 10]
for vidx in vertices:
    graph.set_user_defined_vertex(vidx, False)


renderer = Renderer(graph, graph.vertex_coordinates)
renderer.plot()

graph.smooth_vertices(num_iter=5)
