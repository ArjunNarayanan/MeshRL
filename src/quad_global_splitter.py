import numpy as np
from src.tiler import Tiler, quad_connectivity_representation


class QuadGlobalSplitter:
    def __init__(self, graph: Tiler):
        self.graph = graph
        self.half_edge_aspect_ratios = None
        self.face_degree = 4
        assert all(degree == 4 for fidx, degree in self.graph.face_degrees.items())

    def _ordered_half_edges(self):
        half_edges = []
        for idx, face_index in enumerate(self.graph.face_list()):
            hidx = self.graph.first_face_halfedge(face_index)
            for step in range(self.face_degree):
                half_edges.append(hidx)
                hidx = self.graph.next_half_edge(hidx)
        return half_edges

    def update_half_edge_aspect_ratios(self):
        representation = quad_connectivity_representation(self.graph)
        edges = representation["edges"]
        coords = representation["coordinates"]
        econn = representation["edge connectivity"]

        edge_lengths = np.linalg.norm(coords[edges[:, 1]] - coords[edges[:, 0]], axis=1)
        half_edge_lengths = edge_lengths[econn]

        ar0 = (np.minimum(half_edge_lengths[:, 0], half_edge_lengths[:, 2]) /
               np.maximum(half_edge_lengths[:, 1], half_edge_lengths[:, 3]))
        ar1 = (np.minimum(half_edge_lengths[:, 1], half_edge_lengths[:, 3]) /
               np.maximum(half_edge_lengths[:, 0], half_edge_lengths[:, 2]))
        half_edge_aspect_ratio = np.column_stack([ar0, ar1, ar0, ar1]).ravel()

        half_edges = self._ordered_half_edges()

        self.half_edge_aspect_ratios = dict(zip(half_edges, half_edge_aspect_ratio))

