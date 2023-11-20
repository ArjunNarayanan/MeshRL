from src.polygraph import PolyGraph
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np


def initialize_graph():
    node_ids = list(range(6))
    np.random.shuffle(node_ids)
    face_loops = [node_ids]
    graph = PolyGraph.from_face_loops(face_loops)
    return graph


class HexEnv(gym.Env):
    def __init__(self):
        super().__init__()
