from src.polygraph import PolyGraph
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np


def initialize_graph_and_desired_degree(shuffle):
    node_ids = list(range(6))
    if shuffle:
        np.random.shuffle(node_ids)

    face_loops = [node_ids]
    coords_list = generate_coordinates()
    coords = dict(zip(node_ids, coords_list))

    graph = PolyGraph.from_face_loops(face_loops, vertex_coordinates=coords)
    desired_degree = dict(zip(node_ids, 6 * [3]))

    return graph, desired_degree


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    return coords


class HexEnv(gym.Env):
    def __init__(self, shuffle_idx=True):
        super().__init__()
        self.template_size = 18
        self.max_edge_addition_steps = 3
        self.num_actions_per_halfedge = self.max_edge_addition_steps + 3
        self.max_actions = 10
        self.shuffle_idx = shuffle_idx

        self.interior_vertex_desired_degree = 6
        self.boundary_vertex_desired_degree = 4
        self.face_desired_degree = 3

        self.num_actions = 0

        graph, desired_degree = initialize_graph_and_desired_degree(self.shuffle_idx)
        self.graph = graph
        self.vertex_desired_degree = desired_degree

        self.reward = 0
        self.score = self.global_score()
        self.no_action_reward = -1

        halfedges = self.graph.halfedge_list()
        self._build_template(halfedges)

    def global_score(self):
        vertices = self.graph.vertex_list(tag=False)
        vertex_score = sum(self._vertex_score(vidx) for vidx in vertices)

        faces = self.graph.face_list(tag=False)
        face_score = sum(self._face_score(fidx) for fidx in faces)

        return face_score + vertex_score

    def _face_score(self, fidx):
        face_degree = self.graph.face_degree(fidx)
        return abs(face_degree - self.face_desired_degree)

    def _vertex_score(self, vidx):
        vertex_degree = self.graph.vertex_degree(vidx)
        return abs(vertex_degree - self.vertex_desired_degree[vidx])

    def _halfedge_vertex_score(self, hidx):
        # return 0 if hidx is not a valid halfedge
        # this is useful when updating the template and a halfedge has been deleted by a previous action
        if not self.graph.is_halfedge(hidx):
            return 0

        vidx = self.graph.source_vertex(hidx, tag=False)
        degree = self.graph.vertex_degree(vidx)
        return abs(degree - self.vertex_desired_degree[vidx])

    def _halfedge_face_score(self, hidx):
        if not self.graph.is_halfedge(hidx):
            return 0

        fidx = self.graph.face(hidx)
        degree = self.graph.face_degree(fidx)
        return abs(degree - self.face_desired_degree)

    def _select_template_source(self, halfedges):
        vertex_score = np.array([self._halfedge_vertex_score(hidx) for hidx in halfedges])
        face_score = np.array([self._halfedge_face_score(hidx) for hidx in halfedges])
        total_score = vertex_score + face_score

        max_indices = np.nonzero(total_score == total_score.max())[0]
        template_source_idx = np.random.choice(max_indices)
        return halfedges[template_source_idx]

    def _build_template(self, halfedges):
        source_halfedge = self._select_template_source(halfedges)
        self.index_to_halfedge = self.graph.knn_halfedges(source_halfedge, self.template_size)
        self.halfedge_to_index = {halfedge: idx for idx, halfedge in enumerate(self.index_to_halfedge)}

    def _step_insert_edge(self, hidx, num_steps):
        if self.graph.is_valid_edge_insert(hidx, num_steps):
            original_face_idx = self.graph.face(hidx)
            original_face_degree = self.graph.face_degree(original_face_idx)

            new_face_idx = self.graph.insert_halfedge(hidx, num_steps)

            original_face_new_degree = self.graph.face_degree(original_face_idx)
            new_face_degree = self.graph.face_degree(new_face_idx)
            face_reward = abs(original_face_degree - self.face_desired_degree) - \
                          (abs(original_face_new_degree - self.face_desired_degree) +
                           abs(new_face_degree - self.face_desired_degree))

            source_vertex = self.graph.source_vertex(hidx, tag=False)
            prev_halfedge = self.graph.previous_halfedge(hidx)
            prev_source_vertex = self.graph.source_vertex(prev_halfedge, tag=False)

            source_vertex_degree = self.graph.vertex_degree(source_vertex)
            prev_source_vertex_degree = self.graph.vertex_degree(prev_source_vertex)

            source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]
            prev_source_vertex_desired_degree = self.vertex_desired_degree[prev_source_vertex]
            vertex_reward = (abs(source_vertex_degree - 1 - source_vertex_desired_degree) +
                             abs(prev_source_vertex_degree - 1 - prev_source_vertex_desired_degree)) - \
                            (abs(source_vertex_degree - source_vertex_desired_degree) +
                             abs(prev_source_vertex_degree - prev_source_vertex_desired_degree))
            self.reward = face_reward + vertex_reward
            self.score -= self.reward
        else:
            self.reward = self.no_action_reward

    def _step_insert_vertex(self, hidx):
        if self.graph.halfedge_on_boundary(hidx):
            face_idx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(face_idx)
            new_vertex_idx = self.graph.insert_vertex(hidx, tag=False)
            self.vertex_desired_degree[new_vertex_idx] = self.boundary_vertex_desired_degree
            self.reward = abs(face_degree - self.face_desired_degree) - \
                          (abs(face_degree + 1 - self.face_desired_degree) +
                           abs(2 - self.boundary_vertex_desired_degree))
        else:
            face_idx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(face_idx)

            twin_hidx = self.graph.twin_halfedge(hidx)
            twin_face = self.graph.face(twin_hidx)
            twin_face_degree = self.graph.face_degree(twin_face)

            new_vertex_idx = self.graph.insert_vertex(hidx, tag=False)
            self.vertex_desired_degree[new_vertex_idx] = self.interior_vertex_desired_degree

            self.reward = (abs(face_degree - self.face_desired_degree) +
                           abs(twin_face_degree - self.face_desired_degree)) - \
                          (abs(face_degree + 1 - self.face_desired_degree) +
                           abs(twin_face_degree + 1 - self.face_desired_degree) +
                           abs(2 - self.interior_vertex_desired_degree))

        self.score -= self.reward

    def _step_delete_edge(self, hidx):
        if self.graph.is_valid_delete_halfedge(hidx):
            face = self.graph.face(hidx)
            twin_hidx = self.graph.twin_halfedge(hidx)
            twin_face = self.graph.face(twin_hidx)

            source_vertex = self.graph.source_vertex(hidx, tag=False)
            target_vertex = self.graph.target_vertex(hidx, tag=False)

            current_face_degree = self.graph.face_degree(face)
            twin_face_degree = self.graph.face_degree(twin_face)

            source_vertex_degree = self.graph.vertex_degree(source_vertex)
            target_vertex_degree = self.graph.vertex_degree(target_vertex)

            source_desired_degree = self.vertex_desired_degree[source_vertex]
            target_desired_degree = self.vertex_desired_degree[target_vertex]

            self.graph.delete_halfedge(hidx)

            self.reward = (abs(current_face_degree - self.face_desired_degree) +
                           abs(twin_face_degree - self.face_desired_degree) +
                           abs(source_vertex_degree - source_desired_degree) +
                           abs(target_vertex_degree - target_desired_degree)) - \
                          (abs(current_face_degree + twin_face_degree - 2 - self.face_desired_degree) +
                           abs(source_vertex_degree - 1 - source_desired_degree) +
                           abs(target_vertex_degree - 1 - target_desired_degree))
            self.score -= self.reward
        else:
            self.reward = self.no_action_reward

    def _step_delete_source_vertex(self, hidx):
        if self.graph.is_valid_delete_source_vertex(hidx):
            if self.graph.halfedge_on_boundary(hidx):
                face_idx = self.graph.face(hidx)
                source_vertex = self.graph.source_vertex(hidx, tag=False)

                face_degree = self.graph.face_degree(face_idx)
                source_vertex_degree = self.graph.vertex_degree(source_vertex)
                source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]

                self.graph.delete_source_vertex(hidx)

                self.reward = (abs(face_degree - self.face_desired_degree) +
                               abs(source_vertex_degree - source_vertex_desired_degree)) - \
                              (abs(face_degree - 1 - self.face_desired_degree))
                self.score -= self.reward
                self.vertex_desired_degree.pop(source_vertex)
            else:
                face_idx = self.graph.face(hidx)
                twin_halfedge = self.graph.twin_halfedge(hidx)
                source_vertex = self.graph.source_vertex(hidx, tag=False)
                twin_face = self.graph.face(twin_halfedge)

                face_degree = self.graph.face_degree(face_idx)
                twin_face_degree = self.graph.face_degree(twin_face)
                source_vertex_degree = self.graph.vertex_degree(source_vertex)
                source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]

                self.graph.delete_source_vertex(hidx)

                self.reward = (abs(face_degree - self.face_desired_degree) +
                               abs(twin_face_degree - self.face_desired_degree) +
                               abs(source_vertex_degree - source_vertex_desired_degree)) - \
                              (abs(face_degree - 1 - self.face_desired_degree) +
                               abs(twin_face_degree - 1 - self.face_desired_degree))
                self.vertex_desired_degree.pop(source_vertex)
        # if action is invalid, update reward to no_action_reward
        else:
            self.reward = self.no_action_reward

    def _step_halfedge_action(self, hidx, action):
        assert 0 <= action < self.num_actions_per_halfedge

        if action < self.max_edge_addition_steps:
            self._step_insert_edge(hidx, action + 1)
        elif action == self.max_edge_addition_steps:
            self._step_delete_edge(hidx)
        elif action == self.max_edge_addition_steps + 1:
            self._step_insert_vertex(hidx)
        else:  # delete vertex
            self._step_delete_source_vertex(hidx)


if __name__ == "__main__":
    env = HexEnv(shuffle_idx=False)
