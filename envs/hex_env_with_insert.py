import os
from src.polygraph import PolyGraph
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
from copy import deepcopy
import pickle


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


def get_max_edge_addition_steps():
    return 3


class HexEnv(gym.Env):
    def __init__(
            self,
            template_size,
            max_actions,
            randomize,
            incremental_reward,
            no_action_reward,
            logdir=None
    ):
        super().__init__()
        self.template_size = template_size
        self.max_edge_addition_steps = get_max_edge_addition_steps()
        self.num_actions_per_halfedge = self.get_num_actions_per_halfedge()
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

        # Attributes for excpetion handling
        self.initial_graph = deepcopy(self.graph)
        self.exception_occurred = False
        self.exception_count = 0
        self.action_sequence = []
        if logdir is None:
            logdir = os.getcwd()
        self.logdir = logdir

        halfedges = self.graph.halfedge_list()
        self._build_template(halfedges)

        self._template_boundary_index = -2
        self._geometric_boundary_index = -1

        self.num_features = self.feature_size()

        self.action_space = Discrete(self.num_actions_per_halfedge)
        self.observation_space = Dict(
            {
                "features": Box(low=0, high=4, shape=(self.template_size, self.num_features)),
                "next": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "previous": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "twin": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "mask": Box(low=-np.inf, high=0, shape=(self.template_size * self.num_actions_per_halfedge,)),
                "progress": Box(low=0, high=1, shape=(1,))
            }
        )

    @staticmethod
    def get_num_actions_per_halfedge():
        return get_max_edge_addition_steps() + 3

    @staticmethod
    def feature_size():
        return 5

    @classmethod
    def from_config(cls, config):
        randomize = config.get("randomize", False)
        template_size = config["template_size"]
        max_actions = config["max_actions"]
        incremental_reward = config["incremental_reward"]
        no_action_reward = config.get("no_action_reward", 0)
        logdir = config.get("logdir", None)
        if no_action_reward is None:
            no_action_reward = 0

        return cls(
            template_size,
            max_actions,
            randomize,
            incremental_reward,
            no_action_reward,
            logdir=logdir
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
        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
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

    def _get_action_mask(self):
        total_num_actions = self.template_size * self.num_actions_per_halfedge
        action_mask = np.array([0 if self.is_valid_action(idx) else -np.inf for idx in range(total_num_actions)],
                               dtype=np.float32)
        return action_mask

    def _get_progress(self):
        return np.array([self.num_actions / self.max_actions])

    def _get_obs(self):
        if self.exception_occurred:
            return self._get_blank_obs()
        else:
            obs = {
                "features": self._get_feature_matrix(),
                "next": self._get_next_edges(),
                "previous": self._get_previous_edges(),
                "twin": self._get_twin_edges(),
                "mask": self._get_action_mask(),
                "progress": self._get_progress()
            }
            return obs

    def _get_blank_obs(self):
        features = np.zeros((self.template_size, self.num_features))
        next_edges = np.arange(self.template_size)
        prev_edges = np.arange(self.template_size)
        twin_edges = np.arange(self.template_size)
        mask = np.zeros(self.template_size * self.num_actions_per_halfedge)
        obs = {
            "features": features,
            "next": next_edges,
            "previous": prev_edges,
            "twin": twin_edges,
            "mask": mask,
            "progress": np.array([0])
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

    def _step_halfedge_action(self, halfedge, action):
        assert 0 <= action < self.num_actions_per_halfedge

        self.action_sequence.append((halfedge, action))

        if self.graph.is_halfedge(halfedge):
            if action < self.max_edge_addition_steps:
                self._step_insert_edge(halfedge, action + 1)
            elif action == self.max_edge_addition_steps:
                self._step_delete_edge(halfedge)
            elif action == self.max_edge_addition_steps + 1:
                self._step_insert_vertex(halfedge)
            else:  # delete vertex
                self._step_delete_source_vertex(halfedge)
        else:
            self.reward = self.no_action_reward

    def is_valid_action(self, linear_action_index):
        halfedge, local_action_index = self._linear_action_index_to_halfedge_and_action(linear_action_index)

        if not self.graph.is_halfedge(halfedge):
            return False

        if local_action_index < self.max_edge_addition_steps:
            if not self.graph.is_valid_edge_insert(halfedge, local_action_index + 1):
                return False

        if local_action_index == self.max_edge_addition_steps:
            if not self.graph.is_valid_delete_halfedge(halfedge):
                return False

        if local_action_index == self.num_actions_per_halfedge - 1:
            if not self.graph.is_valid_delete_source_vertex(halfedge):
                return False

        return True

    def reset(self, seed=None, options=None):
        graph, desired_degree = initialize_graph_and_desired_degree(self.randomize)
        self.graph = graph
        self.vertex_desired_degree = desired_degree

        self.initial_graph = deepcopy(self.graph)
        self.action_sequence = []
        self.exception_occurred = False

        self._build_template(self.graph.halfedge_list())

        self.num_actions = 0
        self.score = self.global_score()
        self.reward = 0

        observation = self._get_obs()
        return observation, {"score": self.score}

    def _log_exception(self):
        exception_filename = "except_env_" + str(self.exception_count) + ".pkl"
        self.exception_count += 1
        exception_filepath = os.path.join(self.logdir, exception_filename)
        output_data = {"graph": self.initial_graph, "actions": self.action_sequence}
        with open(exception_filepath, "wb") as output_file:
            pickle.dump(output_data, output_file)

        print("\n\n\tLOGGED EXCEPTED ENV TO : ", exception_filepath, "\n\n\t")

    def _linear_action_index_to_halfedge_and_action(self, linear_action_index):
        hidx = linear_action_index // self.num_actions_per_halfedge
        local_action_index = linear_action_index % self.num_actions_per_halfedge

        if hidx < len(self.index_to_halfedge):
            halfedge = self.index_to_halfedge[hidx]
        else:
            halfedge = None

        return halfedge, local_action_index

    def step(self, linear_action_index):

        if self.num_actions >= self.max_actions:
            print("WARNING : NUM ACTIONS > MAX ACTIONS!!")  # this should not happen

        halfedge, local_action = self._linear_action_index_to_halfedge_and_action(linear_action_index)

        try:
            self._step_halfedge_action(halfedge, local_action)
            # update the template after step:
            halfedges = self.graph.halfedge_list()
            self._build_template(halfedges)
        except Exception as e:
            self.exception_occurred = True
            print("\n\n\tENCOUNTERED ENVIRONMENT EXCPETION\n\n")
            self._log_exception()

        self.num_actions += 1

        terminated = self.is_terminated()
        observation = self._get_obs()

        if not self.incremental_reward:
            if not terminated:
                self.reward = 0
            else:
                self.reward = self.initial_score - self.score

        return observation, self.reward, terminated, False, {"score": self.score}

    def is_terminated(self):
        if self.num_actions >= self.max_actions:
            return True
        if self.score == 0:
            return True
        if self.exception_occurred:
            return True

        return False
