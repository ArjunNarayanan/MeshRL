import unittest
import numpy as np
from src.render import Renderer
from src.tiler import Tiler, refine


def initialize_graph():
    coords = generate_coordinates()
    faces = [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ]
    graph = Tiler.from_face_loops(faces, vertex_coordinates=coords)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


graph = initialize_graph()
renderer = Renderer(graph, graph.vertex_coordinates)
renderer.plot()

refined_graph = refine(graph, 4)
renderer = Renderer(refined_graph, refined_graph.vertex_coordinates)
renderer.plot()