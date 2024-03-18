import numpy as np
from envs.global_angle_env import AngleEnv
from gymnasium.spaces import Box, Dict


class UnnormalizedAngleEnv(AngleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_feature_size():
        return 5

    def _get_length_features(self):
        num_halfedges = len(self.index_to_half_edge)
        features = np.zeros(num_halfedges, dtype=np.float32)
        for idx, hidx in enumerate(self.index_to_half_edge):
            next_hidx = self.graph.next_half_edge(hidx)
            src = self.graph.source_vertex(hidx)
            next_src = self.graph.source_vertex(next_hidx)

            src_coords = self.graph.vertex_coordinate(src)
            next_coords = self.graph.vertex_coordinate(next_src)

            features[idx] = np.linalg.norm(next_coords - src_coords)
        return features

    def _get_feature_matrix(self):
        source_vertices = self.template_vertices
        faces = self.template_faces

        length_features = self._get_length_features()
        vertex_desired_degree = [self.vertex_desired_degree[vidx] for vidx in source_vertices]
        vertex_irregularities = [
            min(self.graph.vertex_degree(vidx), self.vertex_degree_threshold) for vidx in source_vertices
        ]

        face_irregularities = [
            min(self.graph.face_degree(fidx), self.face_degree_threshold) for fidx in faces
        ]

        angle_irregularities = [
            (self.half_edge_angles[hidx]) / self.desired_angle for hidx in self.index_to_half_edge
        ]

        num_half_edges = len(self.index_to_half_edge)

        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        matrix[:num_half_edges, 0] = length_features
        matrix[:num_half_edges, 1] = vertex_desired_degree
        matrix[:num_half_edges, 2] = vertex_irregularities
        matrix[:num_half_edges, 3] = face_irregularities
        matrix[:num_half_edges, 4] = angle_irregularities

        return matrix
