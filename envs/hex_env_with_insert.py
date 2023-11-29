from src.polygraph import PolyGraph
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import torch


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
    def __init__(
            self,
            randomize=True,
            template_size=18,
            no_action_reward=-4,
            max_actions=20,
            incremental_reward=False
    ):
        super().__init__()
        self.template_size = template_size
        self.max_edge_addition_steps = 3
        self.num_actions_per_halfedge = self.max_edge_addition_steps + 3
        self.max_actions = max_actions
        self.randomize = randomize
        self.incremental_reward = incremental_reward

        self.interior_vertex_desired_degree = 6
        self.boundary_vertex_desired_degree = 4
        self.face_desired_degree = 3

        self.num_actions = 0

        graph, desired_degree = initialize_graph_and_desired_degree(self.randomize)
        self.graph = graph
        self.vertex_desired_degree = desired_degree

        self.reward = 0
        self.score = self.global_score()
        self.initial_score = self.score
        self.no_action_reward = no_action_reward

        halfedges = self.graph.halfedge_list()
        self._build_template(halfedges)

        self._template_boundary_index = -2
        self._geometric_boundary_index = -1

        self.action_space = Discrete(self.num_actions_per_halfedge)
        self.observation_space = Dict(
            {
                "features": Box(low=0, high=4, shape=(self.template_size, 5)),
                "next": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "previous": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "twin": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64)
            }
        )

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

        if self.randomize:
            template_source_idx = np.random.choice(max_indices)
        else:
            template_source_idx = max_indices[0]

        return halfedges[template_source_idx]

    def _build_template(self, halfedges):
        source_halfedge = self._select_template_source(halfedges)
        self.index_to_halfedge = self.graph.knn_halfedges(source_halfedge, self.template_size)
        self.halfedge_to_index = {halfedge: idx for idx, halfedge in enumerate(self.index_to_halfedge)}

    def _get_feature_matrix(self):
        matrix = np.zeros((self.template_size, 5))
        for order, hidx in enumerate(self.index_to_halfedge):
            vidx = self.graph.source_vertex(hidx, tag=False)
            vertex_degree = self.graph.vertex_degree(vidx)
            vertex_desired_degree = self.vertex_desired_degree[vidx]
            is_user_defined_vertex = 1.0 if self.graph.is_user_defined_vertex(vidx) else 0.0

            fidx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(fidx)

            matrix[order, :] = [
                vertex_degree / vertex_desired_degree,
                vertex_desired_degree,
                face_degree / self.face_desired_degree,
                self.face_desired_degree,
                is_user_defined_vertex
            ]

        return matrix

    def _get_next_edges(self):
        next_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_halfedge):
            next_hidx = self.graph.next_halfedge(hidx)
            next_idx = self.halfedge_to_index.get(next_hidx, self._template_boundary_index)
            next_edges[idx] = next_idx
        return next_edges

    def _get_previous_edges(self):
        prev_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_halfedge):
            prev_hidx = self.graph.previous_halfedge(hidx)
            prev_idx = self.halfedge_to_index.get(prev_hidx, self._template_boundary_index)
            prev_edges[idx] = prev_idx
        return prev_edges

    def _get_twin_edges(self):
        twin_edges = np.arange(self.template_size)
        for idx, hidx in enumerate(self.index_to_halfedge):
            if self.graph.halfedge_on_boundary(hidx):
                twin_edges[idx] = self._geometric_boundary_index
            else:
                twin_hidx = self.graph.twin_halfedge(hidx)
                twin_idx = self.halfedge_to_index.get(twin_hidx, self._template_boundary_index)
                twin_edges[idx] = twin_idx
        return twin_edges

    def _get_obs(self):
        obs = {
            "features": self._get_feature_matrix(),
            "next": self._get_next_edges(),
            "previous": self._get_previous_edges(),
            "twin": self._get_twin_edges()
        }
        return obs

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

        if self.graph.is_halfedge(hidx):
            if action < self.max_edge_addition_steps:
                self._step_insert_edge(hidx, action + 1)
            elif action == self.max_edge_addition_steps:
                self._step_delete_edge(hidx)
            elif action == self.max_edge_addition_steps + 1:
                self._step_insert_vertex(hidx)
            else:  # delete vertex
                self._step_delete_source_vertex(hidx)
        else:
            self.reward = self.no_action_reward

    def reset(self, seed=None, options=None):
        graph, desired_degree = initialize_graph_and_desired_degree(self.randomize)
        self.graph = graph
        self.vertex_desired_degree = desired_degree
        self._build_template(self.graph.halfedge_list())

        self.num_actions = 0
        self.score = self.global_score()
        self.reward = 0

        observation = self._get_obs()
        return observation, {"score": self.score}

    def step(self, linear_action_index):

        if self.num_actions >= self.max_actions:
            print("WARNING : NUM ACTIONS > MAX ACTIONS!!")  # this should not happen

        halfedge_idx = linear_action_index // self.num_actions_per_halfedge
        local_action_index = linear_action_index % self.num_actions_per_halfedge
        self._step_halfedge_action(halfedge_idx, local_action_index)
        self.num_actions += 1

        # update the template after step:
        halfedges = self.graph.halfedge_list()
        self._build_template(halfedges)

        terminated = self.is_terminated()
        observation = self._get_obs()

        if not self.incremental_reward:
            if not terminated:
                self.reward = 0
            else:
                self.reward = self.initial_score - self.score

        return observation, self.reward, terminated, False, {"score": self.score}

    def is_terminated(self):
        if self.num_actions >= self.max_actions or self.score == 0:
            return True
        else:
            return False


# env = HexEnv(randomize=False, template_size=3)

if __name__ == "__main__":
    env = HexEnv(randomize=False)
    matrix = env._get_feature_matrix()
