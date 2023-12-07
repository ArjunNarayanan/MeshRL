import os
from src.polygraph import PolyGraph
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
from copy import deepcopy
import pickle
import uuid

import numpy as np


def angle_between(v1, v2):
    dotp = v1[0] * v2[0] + v1[1] * v2[1]
    detp = (v1[0] * v2[1] - v2[0] * v1[1])
    angle = np.degrees(np.arctan2(detp, dotp))
    if angle < 0:
        angle = angle + 360
    return angle


def generate_coordinates(polygon_degree, scale=0.5):
    angle = 2 * np.pi / polygon_degree
    angular_increments = angle * np.arange(polygon_degree)
    radii = (1 - scale) + scale * np.random.rand(polygon_degree)
    x_coord = np.cos(angular_increments) * radii
    y_coord = np.sin(angular_increments) * radii
    coords = [[x, y] for x, y in zip(x_coord, y_coord)]
    return coords


def initialize_graph(polygon_degree):
    coordinates = generate_coordinates(polygon_degree)
    node_ids = list(range(polygon_degree))
    face_loop = [node_ids]
    coordinates = dict(zip(node_ids, coordinates))

    graph = PolyGraph.from_face_loops(face_loop, coordinates)
    return graph


from src.render import Renderer

theta = angle_between([1, 0], [-1, 0])
print("Angle : ", theta)

# graph = initialize_graph(10)
# renderer = Renderer(graph, graph.vertex_coordinates)
# renderer.plot()
