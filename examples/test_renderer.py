from envs.regular_polygon_env import RegularPolygonEnv
from src.render import Renderer
import numpy as np
from src.polygraph import PolyGraph

face_loops = [
    [0, 1, 2, 3, 4, 2, 1, 5, 6, 7],
    [2, 4, 3]
]
graph = PolyGraph.from_face_loops(face_loops)
graph.delete_source_vertex(11)