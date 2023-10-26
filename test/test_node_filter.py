from src.polygraph import PolyGraph
import unittest
import numpy as np
import itertools


face_loops = [
    [0,1,2,7,8,9],
    [2,3,4,5,6,7]
]

num_faces = len(face_loops)
num_half_edges = sum((len(l) for l in face_loops))
vertex_ids = set(itertools.chain.from_iterable(face_loops))
num_vertices = len(vertex_ids)


# vertex_connectivity = np.array(
#     [[0, 1], [1, 2], [2, 7], [7, 8], [8, 9], [9, 0], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 2]]
# )
# face_ids = np.array(6 * [0] + 6 * [1])
#
# graph = PolyGraph(num_half_edges, num_vertices, num_faces)
# graph.add_sequential_face_loop(6, 11)
# graph.add_sequential_face_loop(0, 5)
# graph.add_halfedge_to_vertex_edges(vertex_connectivity)
# graph.add_halfedge_to_face_edges(face_ids)
# graph.add_twin_edges(vertex_connectivity)
#
#
# sources = [graph.source_vertex(idx) for idx in range(12)]