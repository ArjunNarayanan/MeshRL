from src.polygraph import PolyGraph
import torch
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np


def initialize_graph():
    node_ids = [0, 1, 2, 3, 4, 5]
    assert len(node_ids) == 6
    face_loops = [node_ids]
    graph = PolyGraph.from_face_loops(face_loops)

    return graph


def env_score(graph: PolyGraph, face_desired_degree: int):
    score = 0
    for face in range(graph.number_of_faces()):
        score += abs(graph.face_degree(face) - face_desired_degree)
    return score


class HexEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(36)
        self.observation_space = Dict(
            {
                "features": Box(low=0, high=2, shape=(36, 4)),
                "next": Box(low=0, high=35, shape=(36,), dtype=np.int64),
                "previous": Box(low=0, high=35, shape=(36,), dtype=np.int64),
                "twin": Box(low=-1, high=35, shape=(36,), dtype=np.int64)
            }
        )

        self.max_actions = 5
        self.face_desired_degree = 3
        self.vertex_desired_degree = 3

        self.graph = initialize_graph()
        self.num_actions = 0
        self.score = env_score()
        self.reward = 0

    def env_score(self):
        score = 0
        for face in range(self.graph.number_of_faces()):
            score += abs(self.graph.face_degree(face) - self.face_desired_degree)
        return score

    def reset(self):
        self.graph = initialize_graph()
        self.num_actions = 0
        self.score = env_score()
        self.reward = 0
