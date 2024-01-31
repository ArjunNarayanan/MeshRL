from envs.angle_env import AngleEnv
import numpy as np
from src.tiler import Tiler
from src.render import Renderer


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 0, 4, 5],
        [0, 5, 4]
    ]
    coords = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, coords)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.2, 0.8],
        [0.8, 0.2]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


graph = initialize_graph()
renderer = Renderer(graph, graph.vertex_coordinates, label_halfedge=True)
renderer.plot()
graph.smooth_vertices(num_iter=10)
renderer.plot()