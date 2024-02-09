import numpy as np
from src.tiler import Tiler, triangle_connectivity_representation


def initialize_graph():
    coords = generate_coordinates()
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 6, 1],
    ]
    graph = Tiler.from_face_loops(faces, vertex_coordinates=coords)
    return graph


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[0, 0],
              [-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(len(coords)), coords))
    return coords


graph = initialize_graph()
representation = triangle_connectivity_representation(graph)

test_coords = generate_coordinates()
coords = representation["coordinates"]