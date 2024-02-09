import numpy as np
from src.tiler import Tiler, quad_connectivity_representation
import numpy as np


class QuadGlobalSplitter:
    def __init__(self, graph: Tiler, max_aspect_ratio=1.5):
        self.graph = graph
        self.half_edge_aspect_ratios = None
        self.face_degree = 4
        assert all(degree == 4 for fidx, degree in self.graph.face_degrees.items())
        self.split_quantile = 0.5
        self.aspect_ratio_threshold = max_aspect_ratio
        self.max_steps_factor = 2
        self.max_split_steps = self.max_steps_factor * self.graph.number_of_half_edges()

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

    def get_global_split_quantile_score(self, hidx):
        # assert self.graph.is_valid_quad_global_split_source(hidx, self.max_split_steps)

        half_edges = self.graph._get_global_line_half_edges(hidx, self.max_split_steps)
        scores = [self.half_edge_aspect_ratios[hidx] for hidx in half_edges]
        quantile_score = np.quantile(scores, self.split_quantile)
        return quantile_score

    def get_max_global_split_score(self):
        scores = []
        candidates = []

        for bidx in self.graph.boundary_half_edge_list():
            hidx = self.graph.twin_half_edge(bidx)
            if self.graph.is_valid_quad_global_split_source(hidx, self.max_split_steps):
                score = self.get_global_split_quantile_score(hidx)
                scores.append(score)
                candidates.append(hidx)

        max_score = np.max(scores)
        max_indices = np.nonzero(scores == max_score)[0]
        max_index = np.random.choice(max_indices)
        candidate = candidates[max_index]

        return max_score, candidate

    def global_split_loop(self, iterations=5, smooth=3):

        for step in range(iterations):
            self.update_half_edge_aspect_ratios()
            score, candidate = self.get_max_global_split_score()
            if candidate and score > self.aspect_ratio_threshold:
                self.graph.global_split_to_boundary(candidate, self.max_split_steps)
                self.graph.smooth_vertices(num_iter=smooth)
            else:
                break
